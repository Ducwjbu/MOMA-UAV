
import numpy as np
from typing import List, Dict, Tuple, Optional
from utils import Configuration
from city import City
from dataclasses import dataclass, field
from numba import jit, prange
from collections import defaultdict, deque

global_env_config = Configuration("config/environment/default.yaml")
global_beam_angle = global_env_config.uav_params['beam_angle']
a = global_env_config.a2g['a']
b = global_env_config.a2g['b']
eta_los = global_env_config.a2g['eta_los']
eta_nlos = global_env_config.a2g['eta_nlos']

class HandoverTracker:
    '''Track handovers and calculate penalties'''
    def __init__(self):
        self.handover_delay_ms = 50.0
        self.handover_failure_rate = 0.01
        self.recent_handovers = {}
        self.ping_pong_threshold = 5.0
        
        # Statistics
        self.total_handovers = 0
        self.ping_pong_handovers = 0

        # Time of each handover still inside the rate window
        self.handover_times = deque()

    def check_handover(self, ue_id, old_uav, new_uav, current_time):
        if old_uav is None or old_uav == new_uav or new_uav is None:
            return {'handover': False, 'delay_penalty_ms': 0.0, 'ping_pong': False}
        
        self.total_handovers += 1
        
        is_ping_pong = False
        if ue_id in self.recent_handovers:
            time_since_last = current_time - self.recent_handovers[ue_id]
            if time_since_last < self.ping_pong_threshold:
                is_ping_pong = True
                self.ping_pong_handovers += 1
        
        self.recent_handovers[ue_id] = current_time
        self.handover_times.append(current_time)

        handover_delay_ms = self.handover_delay_ms
        if is_ping_pong:
            handover_delay_ms *= 1.5
        
        return {
            'handover': True,
            'delay_penalty_ms': handover_delay_ms,
            'packet_loss_prob': self.handover_failure_rate,
            'ping_pong': is_ping_pong
        }
    
    def get_handover_rate(self, current_time, time_window=60.0):
        """Handovers per minute over the last time_window seconds, window end included"""
        while self.handover_times and current_time - self.handover_times[0] > time_window:
            self.handover_times.popleft()
        return len(self.handover_times) / (time_window / 60.0)

@dataclass
class QoSProfile:
    """Quality of Service profile for network slices"""
    min_rate: float  # bps
    max_latency: float  # ms
    min_reliability: float  # percentage

@dataclass
class UAV:
    """UAV entity with position and resource constraints"""
    id: int
    position: np.ndarray  # [x, y, h]
    max_bandwidth: float  # MHz
    max_power: float  # Watts
    current_power: float
    battery_capacity: float  # Joules
    current_battery: float
    velocity_max: float  # m/s
    buffer_size: int  # bits
    energy_used: Dict[str, float] = field(default_factory=dict)  # Joules used in last step
    RBs: List['ResourceBlock'] = None  # List of ResourceBlocks
    beam_angle: float = 60.0  # NEW: Maximum beam angle in degrees (half-angle from vertical)
    beam_direction: np.ndarray = None  # NEW: Beam pointing direction (default: straight down)
    avg_queuing_delay_ms: float = 0.0  # NEW: Average queuing delay at the UAV
    avg_drop_rate: float = 0.0  # NEW: Average packet drop rate at the UAV
    movement_energy: float = 0.0  # NEW: Cumulative movement energy used
    communication_energy: float = 0.0  # NEW: Cumulative communication energy used

    def __post_init__(self):
        if self.RBs is None:
            self.RBs = []
        if self.beam_direction is None:
            self.beam_direction = np.array([0.0, 0.0, -1.0])  # Default: pointing straight down

    def update_position(self, delta_pos: np.ndarray):
        """Update UAV position with velocity constraints"""
        self.position += delta_pos

class LEOSat:
    """LEO Satellite entity"""
    def __init__(self, id: int, position: np.ndarray, coverage_radius: float):
        self.id = id
        self.position = position  # [x, y, h]
        self.coverage_radius = coverage_radius  # meters

@dataclass
class ResourceBlock:
    """Resource Block allocation"""
    id: int
    bandwidth: float  # Hz
    frequency: Optional[float] = 3.5e6  # Frequency in MHz
    allocated_ue_id: int = -1  # User Equipment ID
    allocated_da_id: Optional[int] = -1  # DemandArea ID

@dataclass
class UE:
    """User Equipment entity"""
    id: int
    position: np.ndarray  # [x, y, h]
    slice_type: str  # embb, urllc, mmtc
    assigned_uav: Optional[int] = None
    assigned_da: Optional[int] = None
    assigned_rb: List[ResourceBlock] = None
    throughput: float = 0.0  # bps
    latency_ms: float = 0.0  # ms
    reliability: float = 0.0  # percentage
    throughput_satisfaction = 0.0
    delay_satisfaction = 0.0
    reliability_satisfaction = 0.0
    is_active: bool = True  # NEW: whether UE is currently active
    velocity: np.ndarray = None  # NEW: velocity vector for movement
    traffic_pattern: List[float] = None # The traffic coming to this UE in each T_L period
    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.assigned_rb is None:
            self.assigned_rb = []

@dataclass
class DemandArea:
    """Demand Area for clustering users"""
    id: int
    uav_id: int
    slice_type: str
    user_ids: List[int]
    allocated_bandwidth: float = 2.0
    raw_allocated_action: float = 0.0  # Store raw action before normalization
    center_position: Optional[np.ndarray] = None
    distance_level: str = "Medium"  # New field: "Low", "Medium", "High"
    RB_ids_list: List[int] = None  # List of Resource Block IDs allocated to this Demand Area
    def __post_init__(self):
        if self.center_position is None:
            self.center_position = np.zeros(3)
        if self.RB_ids_list is None:
            self.RB_ids_list = []

@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # Add cache=True!
def calculate_sinr_batch_numba(ue_positions, uav_positions, 
                                uav_powers, serving_indices, noise_power, 
                                path_loss_exponent):
    """Ultra-fast SINR calculation - NOW WITH CACHING"""
    
    n_calculations = ue_positions.shape[0]
    n_uavs = uav_positions.shape[0]
    sinr_db = np.empty(n_calculations, dtype=np.float32)
    
    # Parallel loop
    for i in prange(n_calculations):
        ue_pos = ue_positions[i]
        serving_idx = serving_indices[i]
        
        wavelength = 3e8 / 3.5e9
        factor = wavelength / (4.0 * 3.141592653589793)  # Hardcode pi
        
        signal_power = 0.0
        interference_power = 0.0
        
        for j in range(n_uavs):
            dx = ue_pos[0] - uav_positions[j, 0]
            dy = ue_pos[1] - uav_positions[j, 1]
            dz = ue_pos[2] - uav_positions[j, 2]
            distance = (dx*dx + dy*dy + dz*dz) ** 0.5
            
            path_loss = (factor / distance) ** path_loss_exponent
            rx_power = path_loss * uav_powers[j]
            
            if j == serving_idx:
                signal_power = rx_power
            else:
                angle_to_ue = np.arccos(-dz / distance) * (180.0 / 3.141592653589793)  # in degrees
                if angle_to_ue <= global_beam_angle:  
                    interference_power += rx_power

        sinr = signal_power / (interference_power + noise_power)
        sinr_db[i] = 10.0 * 0.4342944819032518 * np.log(sinr + 1e-10)  # log10(x) = log(x) / log(10)
        
        # Clip
        if sinr_db[i] < -10.0:
            sinr_db[i] = -10.0
        elif sinr_db[i] > 50.0:
            sinr_db[i] = 50.0
    
    return sinr_db

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def calculate_sinr_batch_numba_a2g(ue_positions, uav_positions,
                                    uav_powers, serving_indices, noise_power,
                                    carrier_frequency, interference_powers,
                                    sidelobe_gain):
    """
    Ultra-fast SINR calculation with Al-Hourani A2G path loss model
    
    Args:
        ue_positions: (N, 3) array of UE positions [x, y, z]
        rb_frequencies: (N,) array of RB frequencies (not used, kept for compatibility)
        uav_positions: (M, 3) array of UAV positions [x, y, h]
        uav_powers: (M,) array of power per RB a UAV sends to the UEs it serves
        serving_indices: (N,) array of serving UAV indices
        noise_power: Noise power in Watts
        carrier_frequency: Carrier frequency in Hz
        interference_powers: (M,) array of power per RB a UAV leaks onto another UAV's RBs
        sidelobe_gain: Linear gain of an interferer's beam at UEs outside it

    Returns:
        sinr_db: (N,) array of SINR values in dB
    """
    n_calculations = ue_positions.shape[0]
    n_uavs = uav_positions.shape[0]
    sinr_db = np.empty(n_calculations, dtype=np.float32)
    
    # Pre-compute frequency in MHz for FSPL calculation
    frequency_mhz = carrier_frequency / 1e6
    
    # Constants
    PI = 3.141592653589793
    LOG10_FACTOR = 0.4342944819032518  # 1/ln(10) for log conversion
    
    # Parallel loop over all UEs
    for i in prange(n_calculations):
        ue_pos = ue_positions[i]
        serving_idx = serving_indices[i]
        
        signal_power = 0.0
        interference_power = 0.0
        
        # Loop over all UAVs
        for j in range(n_uavs):
            # Calculate 3D distance
            dx = ue_pos[0] - uav_positions[j, 0]
            dy = ue_pos[1] - uav_positions[j, 1]
            dz = ue_pos[2] - uav_positions[j, 2]
            
            distance_3d = (dx*dx + dy*dy + dz*dz) ** 0.5
            
            # Avoid division by zero
            if distance_3d < 1.0:
                distance_3d = 1.0
            
            # Calculate horizontal distance (2D)
            distance_2d = (dx*dx + dy*dy) ** 0.5
            
            # Calculate elevation angle in degrees
            if distance_2d > 1e-6:
                elevation_deg = np.arctan(uav_positions[j, 2] / distance_2d) * (180.0 / PI)
            else:
                # UE directly below UAV
                elevation_deg = 90.0
            
            # ==========================================
            # A2G PATH LOSS MODEL
            # ==========================================
            
            # 1. LoS probability: P_LoS(θ) = 1/(1 + a·exp(-b(θ - a)))
            exponent = -b * (elevation_deg - a)
            # Clip exponent to avoid overflow
            if exponent > 20.0:
                exponent = 20.0
            elif exponent < -20.0:
                exponent = -20.0
            
            exp_term = np.exp(exponent)
            p_los = 1.0 / (1.0 + a * exp_term)
            
            # 2. Expected excessive path loss (weighted average)
            eta = eta_los * p_los + eta_nlos * (1.0 - p_los)
            
            # 3. Free-space path loss (FSPL) in dB
            # FSPL = 20·log10(d) + 20·log10(f_MHz) - 27.55
            fspl_db = 20.0 * LOG10_FACTOR * np.log(distance_3d) + \
                      20.0 * LOG10_FACTOR * np.log(frequency_mhz) - 27.55
            
            # 4. Total path loss in dB
            path_loss_db = fspl_db + eta
            
            # 5. Convert path loss to linear scale
            # PL_linear = 10^(-PL_dB/10)
            path_loss_linear = 10.0 ** (-path_loss_db / 10.0)
            
            # ==========================================
            # SIGNAL vs INTERFERENCE
            # ==========================================

            if j == serving_idx:
                # This is the serving UAV
                signal_power = path_loss_linear * uav_powers[j]
            else:
                # Inside the interfering UAV's beam at full gain, outside it through the sidelobes
                angle_to_ue = np.arccos(-dz / distance_3d) * (180.0 / PI)
                rx_power = path_loss_linear * interference_powers[j]
                if angle_to_ue <= global_beam_angle:
                    interference_power += rx_power
                else:
                    interference_power += rx_power * sidelobe_gain
        
        # ==========================================
        # SINR CALCULATION
        # ==========================================
        
        sinr = signal_power / (interference_power + noise_power)
        
        # Convert to dB: 10·log10(sinr)
        sinr_db[i] = 10.0 * LOG10_FACTOR * np.log(sinr + 1e-10)
        
        # Clip to realistic range
        if sinr_db[i] < -10.0:
            sinr_db[i] = -10.0
        elif sinr_db[i] > 50.0:
            sinr_db[i] = 50.0
    
    return sinr_db

class NetworkSlicingEnv:
    """
    UAV-based network slicing environment with CTDE support
    """
    def __init__(self, config_path):

        env_config = Configuration(config_path)

        # Load configuration parameters
        self.num_uavs = env_config.system.num_uavs
        self.num_ues = env_config.system.num_ues
        self.service_area = tuple(env_config.system.service_area)  # (width, height)
        self.uav_fly_range_x = tuple(env_config.system.uav_fly_range_x)  # (min_x, max_x)
        self.uav_fly_range_y = tuple(env_config.system.uav_fly_range_y)  # (min_y, max_y)
        self.uav_fly_range_h = tuple(env_config.system.uav_fly_range_h)  # (min_height, max_height)
        self.num_das_per_slice = env_config.system.num_das_per_slice
        self.min_uav_separation = getattr(env_config.system, 'min_uav_separation', 0.0)  # meters, 0 = off

        # Time scale parameters
        self.T_L = env_config.time.T_L  # Long-term period (seconds)
        self.T_S = env_config.time.T_S  # Short-term period (seconds)
        self.current_time = 0.0
        
        # Network slice QoS profiles
        qos = env_config.qos_profiles
        self.qos_profiles = {
            "embb": QoSProfile(min_rate=qos.embb.min_rate, 
                        max_latency=qos.embb.max_latency, 
                        min_reliability=qos.embb.min_reliability),   # eMBB
            "urllc": QoSProfile(min_rate=qos.urllc.min_rate, 
                        max_latency=qos.urllc.max_latency, 
                        min_reliability=qos.urllc.min_reliability),   # URLLC
            "mmtc": QoSProfile(min_rate=qos.mmtc.min_rate, 
                        max_latency=qos.mmtc.max_latency, 
                        min_reliability=qos.mmtc.min_reliability)     # mMTC
        }
        self.qos_weights = env_config.qos_weights  # Weights for throughput, delay, reliability
        
        # Slice-related parameters
        self.slice_weights = env_config.slicing.slice_weights  # Weights for each slice type
        self.slice_probs = env_config.slicing.slice_probabilities  # Probabilities for each slice type

        # Channel parameters
        self.carrier_frequency = env_config.channel.carrier_frequency  # Hz

        # Resource block parameters
        self.rb_bandwidth = env_config.channel.rb_bandwidth  # Hz (12 subcarriers x 30 kHz)
        self.total_bandwidth = env_config.channel.total_bandwidth  # Hz per UAV
        self.total_rbs = int(round(self.total_bandwidth / self.rb_bandwidth))
        self.carrier_frequency = env_config.channel.carrier_frequency  # Hz
        self.shannon_attenuation = env_config.channel.shannon_attenuation
        self.max_spectral_efficiency = env_config.channel.max_spectral_efficiency  # bps/Hz
        # Mogensen et al. 2007 Eq. 3: SE = att * log2(1 + SINR / SNR_eff); 0 dB (absent) is plain attenuated Shannon
        self.snr_efficiency = 10.0 ** (getattr(env_config.channel, 'snr_efficiency_db', 0.0) / 10.0)
        # Below the lowest CQI's efficiency no MCS decodes: the rate is 0 (absent: no floor)
        self.min_spectral_efficiency = getattr(env_config.channel, 'min_spectral_efficiency', 0.0)

        # Noise power per RB (Watts)
        noise_dbm = (env_config.channel.noise_psd_dbm_hz + env_config.channel.noise_figure_db
                     + 10.0 * np.log10(self.rb_bandwidth))
        self.noise_power = 10.0 ** ((noise_dbm - 30.0) / 10.0)

        # Wireless backhaul from each UAV to one ground gateway; without the section it is unlimited
        self.backhaul = getattr(env_config, 'backhaul', None)
        self.backhaul_load = {}  # {uav_id: carried traffic / backhaul capacity}
        self.backhaul_capacity = {}  # {uav_id: bps}
        if self.backhaul is not None:
            self.backhaul_gateway = np.array(self.backhaul.gateway_position, dtype=np.float64)
            backhaul_noise_dbm = (env_config.channel.noise_psd_dbm_hz + env_config.channel.noise_figure_db
                                  + 10.0 * np.log10(self.backhaul.bandwidth))
            self.backhaul_noise_power = 10.0 ** ((backhaul_noise_dbm - 30.0) / 10.0)
        print(f"Total Resource Blocks per UAV: {self.total_rbs}")

        # Handover tracker
        self.handover_tracker = HandoverTracker()

        # UAV parameters
        self.uav_params = env_config.uav_params  # UAV parameters like max power, battery, etc.
        self.uav_beam_angle = getattr(env_config.uav_params, 'beam_angle', 60.0)
        # Without the key there are no sidelobes: UEs outside a beam get no interference from it
        sidelobe_db = getattr(env_config.uav_params, 'sidelobe_attenuation_db', None)
        self.sidelobe_gain = 0.0 if sidelobe_db is None else 10.0 ** (-sidelobe_db / 10.0)
        # A2G parameters of this config (the numba path still reads default.yaml's at import)
        self.a2g = env_config.a2g
        # 3GPP TR 38.901 antenna element pointing down. With it, a UE associates to the UAV with the
        # strongest reference signal and is covered above a minimum SNR, instead of a hard beam cone
        self.antenna = getattr(env_config.uav_params, 'antenna', None)
        # Log-normal shadowing of TR 36.777 (UMa-AV), with the LoS state drawn per UE-UAV link and step
        self.shadowing = getattr(env_config.channel, 'shadowing', None)
        self.channel_draws = {}  # {ue_id: (uniform for the LoS state, standard normal)}, one per UAV
        # Explicit ITU-R P.1410 city (city.py), regenerated each episode. With it, LoS is geometric, UEs inside
        # a footprint are indoor (O2I loss), and shadowing is correlated in space. Absent: the statistical model
        self.city_params = getattr(env_config, 'city', None)
        self.city = None
        self.o2i = getattr(env_config.channel, 'o2i', None)
        self.fast_fading = getattr(env_config.channel, 'fast_fading', None)
        self._se_table = self._build_fading_table() if self.fast_fading is not None else None
        # Horizontal wind of Dai et al. (arXiv:2304.06909), drawn once per episode. Absent: no wind
        self.wind_params = getattr(env_config, 'wind', None)
        self.wind_ref = np.zeros(2)  # wind vector at the reference height, m/s (x east, y north)
        self.wind_ceiling = None  # highest altitude where the wind stays within the UAV's limit
        self.wind_rejections = 0  # episodes redrawn because the wind exceeded the limit even at the lowest altitude
        self.energy_models = env_config.energy_models
        self.uav_buffer_size = env_config.uav_params.buffer_size  # bits
        self.velocity_max_vertical = env_config.uav_params.velocity_max_vertical  # m/s, climb
        self.velocity_max_descent = getattr(env_config.uav_params, 'velocity_max_descent', self.velocity_max_vertical)
        self.processing_delay_ms = env_config.uav_params.processing_delay_ms
        self.min_power_ratio = 0.05  # Lowest transmit power, as a fraction of max_power
        self._init_energy_bounds()

        # UE dynamics parameters
        self.ue_id_pool = set()  # Pool of available IDs for reuse
        self.next_ue_id = 0  # Only used when pool is empty
        self.ue_dynamics = env_config.ue_dynamics  # UE dynamics parameters
        self.ue_arrival_rate = self.ue_dynamics.arrival_rate  # New UEs per second
        # Probability of a UE leaving per T_L, from exponential session durations
        self.ue_departure_rate = 1.0 - np.exp(-self.T_L / self.ue_dynamics.session_duration_avg)
        self.ue_max_initial_velocity = self.ue_dynamics.max_initial_velocity  # m/s
        self.max_ues = int(self.num_ues * self.ue_dynamics.max_ues_multiplier)  # Maximum UEs allowed
        self.change_direction_prob = self.ue_dynamics.change_direction_prob  # Probability of changing direction

        self.hotspots = []
        self.hotspot_attraction_strength = 0.6
        self.hotspot_spawn_rate = 0.05  # Probability per T_L period (5% chance)
        self.hotspot_max_count = 4  # Maximum concurrent hotspots
        self.hotspot_min_lifetime = 900.0  # seconds
        self.hotspot_max_lifetime = 1800.0  # seconds

        #Traffic model
        self.traffic_patterns = env_config.traffic_patterns
        tp = self.traffic_patterns
        # Mean mMTC packet: Pareto(shape, min) payload capped at payload_cap (TR 45.820 Annex E), plus overhead.
        # E[min(X, c)] = E[X] - E[(X - c)+] with E[X] = a m / (a - 1), E[(X - c)+] = m^a / ((a - 1) c^(a - 1))
        a, m, c = tp.mmtc.pareto_shape, tp.mmtc.payload_min, tp.mmtc.payload_cap
        mmtc_payload = a * m / (a - 1.0) - m ** a / ((a - 1.0) * c ** (a - 1.0))
        self.packet_bytes = {'embb': tp.embb.packet_size, 'urllc': tp.urllc.packet_size,
                             'mmtc': mmtc_payload + tp.mmtc.overhead}
        # Reports per second per device: mix of periodicities
        self.mmtc_report_rate = float(np.dot(tp.mmtc.report_interval_probs, 1.0 / np.array(tp.mmtc.report_intervals_s)))
        # Road-side units (URLLC): fixed, at intersections of a road grid (TR 36.885 Annex A.1.2-A.1.3)
        self.rsu_params = getattr(env_config, 'rsu', None)

        # Distance thresholds for forming Demand Areas
        self.da_distance_thresholds = env_config.da_distance_thresholds

        # SINR thresholds for DA classification (in dB)
        self.sinr_thresholds = env_config.sinr_thresholds  

        # Reward weights
        self.reward_weights = env_config.reward_weights
        # Allocation metric of Jain's index: 'min_rate' (throughput / slice min_rate, uncapped) or
        # 'fraction_of_demand' (Jain et al. 1984: throughput / offered load, capped at 1)
        self.fairness_metric = getattr(env_config, 'fairness_metric', 'min_rate')

        # Statistics tracking
        self.stats = {
            'arrivals': 0,
            'departures': 0,
            'handovers': 0,
            'reward': 0.0,
            'qos': 0.0,
            'energy': 0.0,
            'fairness': 0.0,
        }
        
        # Initialize entities
        self.reset()
        self._print_statistics()

        # Add these new attributes for caching
        self.sinr_cache = {}
        self.sinr_cache_valid = False

        self.delay_cache = {}
        self.delay_cache_valid = False

        self.per_cache = {}
        self.per_cache_valid = False

        self.throughput_cache = {}
        self.throughput_cache_valid = False
        
        # Optional: track cache performance
        self.cache_hits = 0
        self.cache_misses = 0
        
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment to initial state"""
        # A new city and a new wind every episode, from the seeded global RNG
        if self.city_params is not None:
            cp = self.city_params
            self.city = City(self.service_area, cp['built_up_ratio'], cp['buildings_per_km2'], cp['height_scale'])
        if self.wind_params is not None:
            self._draw_wind()

        # Initialize UAVs at random positions within service area
        self.uavs = {}
        for i in range(self.num_uavs):
            x = np.random.uniform(self.uav_fly_range_x[0], self.uav_fly_range_x[1])
            y = np.random.uniform(self.uav_fly_range_y[0], self.uav_fly_range_y[1])
            h = np.random.uniform(self.uav_fly_range_h[0], self.uav_fly_range_h[1])
            
            self.uavs[i] = UAV(
                id=i,
                position=np.array([x, y, h]),
                max_bandwidth=self.total_bandwidth,
                max_power=self.uav_params.max_power,  # Watts
                current_power=self.uav_params.initial_power,
                battery_capacity=self.uav_params.battery_capacity,  # Joules
                # current_battery=self.uav_params.initial_battery,  # Start fully charged
                current_battery=self.uav_params.battery_capacity,  # Start with zero power
                velocity_max=self.uav_params.velocity_max,  # m/s
                buffer_size=self.uav_buffer_size,  # bits
                energy_used={"movement": 0.0, "transmission": 0.0},
                RBs=[ResourceBlock(id=i, bandwidth = self.rb_bandwidth, frequency = self.carrier_frequency) for i in range(self.total_rbs)],
                beam_angle=self.uav_beam_angle,  # NEW
                beam_direction=np.array([0.0, 0.0, -1.0])  # NEW: Default pointing down
            )

        if self._separation_active():
            positions = self._separate_uavs(np.array([uav.position for uav in self.uavs.values()]))
            for uav, pos in zip(self.uavs.values(), positions):
                uav.position = pos
        for uav in self.uavs.values():
            uav.position = self._apply_height_bounds(uav.position)

        # Episodes start with two active hotspots, so UEs are clustered from step 0
        self.hotspots = []
        for _ in range(2):
            hotspot = self._spawn_new_hotspot()
            hotspot['age'] = hotspot['lifetime'] * 0.2
            hotspot['current_strength'] = hotspot['max_strength']
            hotspot['state'] = 'active'

        self.handover_tracker = HandoverTracker()

        # Initialize UEs with random positions and slice types
        self.ues = {}
        self.ue_id_pool = set()
        self.next_ue_id = 0

        for i in range(self.num_ues):
            self._add_new_ue()
        self._add_rsus()

        # Initialize demand areas
        self.demand_areas = {}
        self.da_counter = 0
        self.backhaul_load = {uav_id: 0.0 for uav_id in self.uavs}
        self.channel_draws = {}
        self._sf_uav_positions = None
        self._update_channel_state()

        # Perform initial UAV-UE association and DA formation
        self._associate_ues_to_uavs()
        self._form_demand_areas()
        
        # Get initial observations for each UAV
        observations = self._get_observations()
        
        self.current_time = 0.0
        return observations

    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for each UAV agent
        Logic:
        Each UAV agent combines:
        - The UAV's state (5 dims):
            + position (x y h)
            + power level (normalized)
            + battery level (normalized)
        - The UAV's demand area info, 9 DAs (slice x distance level) in fixed order:
            + number of users (share of the UAV's UEs): 1 dim
            + with one UAV only, slice type and distance level one-hots: 6 constant dims
        - The UAV's handover state (4 dims):
            - recent handover rate (normalized)
            - ping-pong rate (normalized)
            - number of UEs currently in handover (normalized)
            - average handover penalty being experienced by UEs (normalized)
        - The UAV's surrounding condition (8 dims):
            - share of UEs to each direction (N, E, S, W)
            - share of other UAVs to each direction (N, E, S, W)
        - With more than one UAV:
            - share of uncovered UEs to each direction (4 dims)
            - each other UAV in id order: dx, dy, dh over the fly range, power ratio (4 dims each)
        - With a backhaul section in the config: the UAV's backhaul load, capped at 2 and halved (1 dim)

        => One UAV: 5 + 63 + 4 + 8 = 80 dims
        => N > 1 UAVs: 5 + 9 + 4 + 8 + 4 + 4 * (N - 1) = 30 + 4 * (N - 1) dims
        => Plus 1 with a backhaul section
        """
        observations = {}

        num_total_ues = max(1, len([ue for ue in self.ues.values() if ue.is_active]))
        slice_types = ["embb", "urllc", "mmtc"]
        distance_levels = ["Near", "Medium", "Far"]

        for uav_id, uav in self.uavs.items():
            obs = []

            # ============================================================
            # 1. UAV State (5 dims)
            # ============================================================
            norm_x = (uav.position[0] - self.uav_fly_range_x[0]) / (self.uav_fly_range_x[1] - self.uav_fly_range_x[0])
            norm_y = (uav.position[1] - self.uav_fly_range_y[0]) / (self.uav_fly_range_y[1] - self.uav_fly_range_y[0])
            norm_h = (uav.position[2] - self.uav_fly_range_h[0]) / (self.uav_fly_range_h[1] - self.uav_fly_range_h[0])
            obs.extend([norm_x, norm_y, norm_h])  # 3
            obs.append(uav.current_power / uav.max_power)  # 1
            obs.append(uav.current_battery / uav.battery_capacity)  # 1

            # ============================================================
            # 2. Demand Area Info (9 DAs × 7 features = 63 dims). So far 5 + 63 = 68
            # ============================================================
            # Fixed order: embb-Near, embb-Medium, embb-Far, urllc-Near, ...
            for slice_type in slice_types:  # 3 types
                for distance_level in distance_levels:  # 3 levels
                    da = next((da for da in self.demand_areas.values()
                            if da.uav_id == uav_id and 
                            da.slice_type == slice_type and 
                            da.distance_level == distance_level), None)
                    
                    # Number of users (normalized) (1 dim) ---
                    total_uav_ues = len([ue for ue in self.ues.values() 
                                    if ue.assigned_uav == uav_id and ue.is_active])
                    num_users = len(da.user_ids) / max(1, total_uav_ues) if da else 0.0
                    obs.append(num_users)
                    
                    if self.num_uavs == 1:
                        # Slice type one-hot (3 dims)
                        obs.extend([1.0 if slice_type == st else 0.0 for st in slice_types])

                        # Distance level one-hot (3 dims)
                        obs.extend([1.0 if distance_level == st else 0.0 for st in distance_levels])

            # ============================================
            # 3. NEW: Handover State Features (4 dims). So far 5 + 63 + 4 = 72
            # ============================================
            # Recent handover rate (handovers per minute) (1 dim)
            handover_rate = self.handover_tracker.get_handover_rate(self.current_time, time_window=self.T_L)
            obs.append(handover_rate / 50.0)  # Normalize (assume max ~10/min)
            
            # Ping-pong rate (1 dim)
            total_ho = max(self.handover_tracker.total_handovers, 1)
            ping_pong_rate = self.handover_tracker.ping_pong_handovers / total_ho
            obs.append(ping_pong_rate)
            
            # Number of UEs currently in handover (1 dim)
            ues_in_handover = sum(1 for ue in self.ues.values()
                                if ue.assigned_uav == uav_id and 
                                getattr(ue, 'handover_penalty_ms', 0) > 0)
            obs.append(ues_in_handover / 10.0)  # Normalize
            
            # Average handover penalty being experienced by UEs (1 dim)
            # Average handover penalty being experienced by UEs (1 dim)
            ho_penalties = [getattr(ue, 'handover_penalty_ms', 0) 
                            for ue in self.ues.values()
                            if ue.assigned_uav == uav_id and ue.is_active]
            avg_ho_penalty = np.mean(ho_penalties) if ho_penalties else 0.0  # Fix: handle empty list
            obs.append(avg_ho_penalty / 50.0)  # Normalize by 50ms


            # ============================================================
            # 3. Surrounding Condition (8 dims). So far 5 + 63 + 4 + 8 = 80
            # ============================================================
            # UE distribution in 4 directions
            directions = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
            for ue in self.ues.values():
                if not ue.is_active:
                    continue
                dx = ue.position[0] - uav.position[0]
                dy = ue.position[1] - uav.position[1]
                
                # Quadrant assignment
                if dx > 0:
                    directions['E' if dy > 0 else 'S'] += 1
                else:
                    directions['W' if dy > 0 else 'N'] += 1
            
            for dir in ['N', 'E', 'S', 'W']:
                obs.append(directions[dir] / num_total_ues)  # 4
            
            # Interference from other UAVs in 4 directions
            interference_dirs = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
            for other_id, other_uav in self.uavs.items():
                if other_id == uav_id:
                    continue
                dx = other_uav.position[0] - uav.position[0]
                dy = other_uav.position[1] - uav.position[1]
                
                if dx > 0:
                    interference_dirs['E' if dy > 0 else 'S'] += 1
                else:
                    interference_dirs['W' if dy > 0 else 'N'] += 1
            
            for dir in ['N', 'E', 'S', 'W']:
                obs.append(interference_dirs[dir] / max(1, self.num_uavs - 1))  # 4

            if self.num_uavs > 1:
                # Uncovered UEs in 4 directions: where this UAV could add coverage
                uncovered_dirs = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
                for ue in self.ues.values():
                    if not ue.is_active or ue.assigned_uav is not None:
                        continue
                    dx = ue.position[0] - uav.position[0]
                    dy = ue.position[1] - uav.position[1]

                    if dx > 0:
                        uncovered_dirs['E' if dy > 0 else 'S'] += 1
                    else:
                        uncovered_dirs['W' if dy > 0 else 'N'] += 1

                for dir in ['N', 'E', 'S', 'W']:
                    obs.append(uncovered_dirs[dir] / num_total_ues)  # 4

                # The other UAVs, in id order
                for other_id in sorted(self.uavs.keys()):
                    if other_id == uav_id:
                        continue
                    other_uav = self.uavs[other_id]
                    obs.append((other_uav.position[0] - uav.position[0]) / (self.uav_fly_range_x[1] - self.uav_fly_range_x[0]))
                    obs.append((other_uav.position[1] - uav.position[1]) / (self.uav_fly_range_y[1] - self.uav_fly_range_y[0]))
                    obs.append((other_uav.position[2] - uav.position[2]) / (self.uav_fly_range_h[1] - self.uav_fly_range_h[0]))
                    obs.append(other_uav.current_power / other_uav.max_power)  # 4 per other UAV

            if self.backhaul is not None:
                obs.append(min(self.backhaul_load[uav_id], 2.0) / 2.0)

            if self.wind_params is not None:
                # Wind at the UAV's altitude, which the flight controller estimates
                obs.extend(self._wind_at(uav.position[2]) / self.wind_params['max_speed'])

            observations[uav_id] = np.array(obs, dtype=np.float32)
            # print(f"UAV {uav_id} observations: {obs}")

            # Sanity check
            expected_dims = 80 if self.num_uavs == 1 else 30 + 4 * (self.num_uavs - 1)
            expected_dims += self.backhaul is not None
            expected_dims += 2 * (self.wind_params is not None)
            assert len(obs) == expected_dims, f"Expected {expected_dims} dims, got {len(obs)}"

        return observations

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], float, bool, Dict]:
        """Execute actions and return next observations, reward, done, info"""

        # Parse actions for each UAV: all moves are decided before any energy is counted
        new_positions = {}
        for uav_id, action in actions.items():
            # print(f"UAV {uav_id} action: {action}")
            uav = self.uavs[uav_id]
            
            # Action format: [delta_x, delta_y, delta_h, power, bandwidth_allocation_per_da...]
            # Position update (constrained by velocity)
            delta_xy = action[:2] * uav.velocity_max * self.T_L
            max_delta_xy = uav.velocity_max * self.T_L
            delta_xy = delta_xy * min(1.0, max_delta_xy / max(np.linalg.norm(delta_xy), 1e-9))
            if self.wind_params is not None:
                # The commanded ground velocity is cut back so that the airspeed stays within velocity_max
                delta_xy = self._limit_airspeed(delta_xy / self.T_L, self._wind_at(uav.position[2]),
                                                uav.velocity_max) * self.T_L
            vertical_rate = self.velocity_max_vertical if action[2] > 0 else self.velocity_max_descent
            delta_h = action[2] * vertical_rate * self.T_L
            new_pos = uav.position + np.array([delta_xy[0], delta_xy[1], delta_h])
            # Constrain to service area
            new_pos[0] = np.clip(new_pos[0], self.uav_fly_range_x[0], self.uav_fly_range_x[1])
            new_pos[1] = np.clip(new_pos[1], self.uav_fly_range_y[0], self.uav_fly_range_y[1])
            new_pos[2] = np.clip(new_pos[2], self.uav_fly_range_h[0], self.uav_fly_range_h[1])
            new_positions[uav_id] = new_pos

        if self._separation_active():
            # UAVs without an action in this call hold their position
            all_positions = np.array([new_positions.get(uav_id, uav.position) for uav_id, uav in self.uavs.items()])
            separated = self._separate_uavs(all_positions)
            for uav_id, pos in zip(self.uavs.keys(), separated):
                if uav_id in actions:
                    new_positions[uav_id] = pos
                else:
                    self.uavs[uav_id].position = pos

        # Rooftop clearance and the wind ceiling, after the separation push
        for uav_id in new_positions:
            new_positions[uav_id] = self._apply_height_bounds(new_positions[uav_id])

        for uav_id, action in actions.items():
            uav = self.uavs[uav_id]
            new_pos = new_positions[uav_id]

            # Calculate movement energy cost
            v_x = (new_pos[0] - uav.position[0]) / self.T_L
            v_y = (new_pos[1] - uav.position[1]) / self.T_L
            v_z = (new_pos[2] - uav.position[2]) / self.T_L

            if self.wind_params is not None:
                wind = self._wind_at((uav.position[2] + new_pos[2]) / 2.0)
                airspeed = np.hypot(v_x - wind[0], v_y - wind[1])
                movement_energy = self._propulsion_power(np.hypot(v_x, v_y), v_z, airspeed) * self.T_L
            else:
                movement_energy = self._propulsion_power(np.hypot(v_x, v_y), v_z) * self.T_L

            uav.position = new_pos

            # Power update
            # print("Action power component:", action[3])
            uav.current_power = np.clip(action[3], self.min_power_ratio, 1) * uav.max_power

            # Energy consumption
            transmission_energy = self._radio_power(uav.current_power) * self.T_L
            uav.energy_used = {"movement": movement_energy, "transmission": transmission_energy}
            uav.current_battery -= sum(uav.energy_used.values())
            uav.current_battery = max(0, uav.current_battery)


        # Invalidate cache BEFORE making changes
        self._invalidate_sinr_cache()
        self._invalidate_throughput_cache()
        

        # Update UE dynamics
        self._update_ue_dynamics()

        # New shadowing and LoS states for this step, shared by association and SINR
        self.channel_draws = {}
        self._update_channel_state()

        # Update associations and DAs (every long-term period)
        self._associate_ues_to_uavs()
        self._form_demand_areas()

        # Update UEs traffic patterns
        self._update_ue_traffic_patterns()

        # Apply bandwidth allocation actions to DAs
        for uav_id, action in actions.items():
            uav_das = [da for da in self.demand_areas.values() if da.uav_id == uav_id]
            if len(uav_das) > 0:
                # Normalized bandwidth allocation
                bandwidth_actions = action[4:4+len(uav_das)]
                
                for i, da in enumerate(uav_das):
                    da.raw_allocated_action = bandwidth_actions[i]
                    da.allocated_bandwidth = bandwidth_actions[i] * self.uavs[uav_id].max_bandwidth

        # Bandwidth allocation to DAs
        self._allocate_rbs_fairly()

        # Update SINR, throughput, delay, reliability caches
        self._update_sinr_cache()
        self._update_throughput_cache()

        # Update queuing statistics at UAVs (the delay cache reads them)
        self._update_uavs_queuing_stats()

        self._update_delay_cache()
        self._update_per_cache()

        # Calculate rewards
        # print("Is throughput cache valid?", self.throughput_cache_valid)
        reward, reward_breakdown = self.calculate_global_reward()

        # Vector reward for MORL (backward compatible, added to info), from the same computation
        reward_vector = np.array([reward_breakdown['qos_reward'], -reward_breakdown['energy_penalty'],
                                  reward_breakdown['fairness_reward']], dtype=np.float32)

        # Update time
        self.current_time += self.T_L
        
        # Check termination conditions
        done = any(uav.current_battery <= 0 for uav in self.uavs.values())

        # Get new observations
        observations = self._get_observations()

        self.stats['reward'] = reward
        self.stats['qos'] = reward_breakdown['qos_reward']
        self.stats['energy'] = reward_breakdown['energy_penalty']
        self.stats['fairness'] = reward_breakdown['fairness_reward']
        self.stats['avg_throughput_sat'] = reward_breakdown.get('avg_throughput_sat', 0.0)
        self.stats['avg_delay_sat'] = reward_breakdown.get('avg_delay_sat', np.inf)
        self.stats['avg_reliability_sat'] = reward_breakdown.get('avg_reliability_sat', 0.0)

        active_ues = [ue for ue in self.ues.values() if ue.is_active]
        info = {
            'qos_satisfaction': reward_breakdown['qos_reward'],
            'energy_usage_level': reward_breakdown['energy_penalty'],
            'fairness_level': reward_breakdown['fairness_reward'],
            'active_ues': len(active_ues),
            'ue_arrivals': self.stats['arrivals'],
            'ue_departures': self.stats['departures'],
            # MORL: vector reward [qos, -energy, fairness] for multi-objective agents
            'reward_vector': reward_vector,
            # Team diagnostics. min_uav_distance is inf when there is only one UAV
            'coverage': sum(ue.assigned_uav is not None for ue in active_ues) / max(1, len(active_ues)),
            'handovers': self.stats['handovers'],
            'min_uav_distance': self._min_uav_distance(),
            # Share of UAVs whose traffic exceeds their backhaul
            'backhaul_limited': sum(load > 1.0 for load in self.backhaul_load.values()) / self.num_uavs,
            # Wind speed at the reference height (0 without wind)
            'wind_speed': float(np.linalg.norm(self.wind_ref)),
        }
        
        return observations, reward, done, info

    # ============================================
    # WIND AND FLIGHT LIMITS
    # ============================================

    def _draw_wind(self):
        """
        Dai et al. (arXiv:2304.06909) Eq. 1-3: speed at the reference height from Weibull(scale, shape), power
        law in height, direction from von Mises. The config gives the direction the wind blows FROM, clockwise
        from north (as wind atlases do). A wind above the UAV's limit even at the lowest altitude is a no-fly
        day: it is redrawn and counted in wind_rejections
        """
        w = self.wind_params
        h_min = self.uav_fly_range_h[0]
        while True:
            v_ref = w['weibull_scale'] * np.random.weibull(w['weibull_shape'])
            if 'direction_frequencies' in w:
                # Measured wind rose: equal sectors clockwise from north, uniform within the drawn sector
                freq = np.asarray(w['direction_frequencies'], dtype=np.float64)
                width = 360.0 / len(freq)
                sector = np.random.choice(len(freq), p=freq / freq.sum())
                from_deg = sector * width + np.random.uniform(-width / 2.0, width / 2.0)
            else:
                from_deg = np.degrees(np.random.vonmises(np.radians(w['direction_from_deg']), w['direction_kappa']))
            if v_ref * (h_min / w['reference_height']) ** w['exponent'] <= w['max_speed']:
                break
            self.wind_rejections += 1
        to_rad = np.radians(from_deg + 180.0)  # bearing it blows towards
        self.wind_ref = v_ref * np.array([np.sin(to_rad), np.cos(to_rad)])
        if v_ref > 0:
            self.wind_ceiling = w['reference_height'] * (w['max_speed'] / v_ref) ** (1.0 / w['exponent'])
        else:
            self.wind_ceiling = np.inf

    def _wind_at(self, height: float) -> np.ndarray:
        """Horizontal wind vector at a height, m/s; zero without a wind section"""
        if self.wind_params is None:
            return np.zeros(2)
        w = self.wind_params
        return self.wind_ref * (max(height, 1.0) / w['reference_height']) ** w['exponent']

    @staticmethod
    def _limit_airspeed(ground_velocity: np.ndarray, wind: np.ndarray, max_airspeed: float) -> np.ndarray:
        """Shortest cut of the commanded ground velocity, same heading, with |v - wind| <= max_airspeed"""
        speed = np.linalg.norm(ground_velocity)
        if speed < 1e-9 or np.linalg.norm(ground_velocity - wind) <= max_airspeed:
            return ground_velocity
        u = ground_velocity / speed
        along = u @ wind
        reach = along + np.sqrt(max(along ** 2 - wind @ wind + max_airspeed ** 2, 0.0))
        return u * min(speed, max(reach, 0.0))

    def _apply_height_bounds(self, position: np.ndarray) -> np.ndarray:
        """
        Keep the UAV above max(min height, roof + clearance) at its spot, and below the wind ceiling (where the
        wind reaches the UAV's limit). Rooftop clearance wins if the two conflict
        """
        lower, upper = self.uav_fly_range_h
        if self.city is not None:
            roof = self.city.roof_height(position[np.newaxis, :2])[0]
            lower = max(lower, roof + self.city_params['roof_clearance'])
        if self.wind_ceiling is not None:
            upper = min(upper, self.wind_ceiling)
        position = position.copy()
        position[2] = max(min(position[2], upper), lower)
        return position

    # ============================================
    # UAV SEPARATION METHODS
    # ============================================

    def _separation_active(self) -> bool:
        return self.num_uavs > 1 and self.min_uav_separation > 0

    def _separate_uavs(self, positions: np.ndarray) -> np.ndarray:
        """
        Push UAV pairs apart until every 3D distance reaches min_uav_separation, staying
        in the fly ranges. Deterministic, so it uses no random numbers: UAVs at the same
        point split along a horizontal direction set by their ids.
        """
        lower = np.array([self.uav_fly_range_x[0], self.uav_fly_range_y[0], self.uav_fly_range_h[0]], dtype=float)
        upper = np.array([self.uav_fly_range_x[1], self.uav_fly_range_y[1], self.uav_fly_range_h[1]], dtype=float)
        margin = 1e-6  # A pair stopped at the fly range closes half its gap per pass; this makes it finish
        positions = positions.astype(float)
        n = len(positions)
        for _ in range(100):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    delta = positions[j] - positions[i]
                    distance = np.linalg.norm(delta)
                    if distance >= self.min_uav_separation:
                        continue
                    if distance > 1e-9:
                        direction = delta / distance
                    else:
                        angle = 2.0 * np.pi * (i + 1) * (j + 1) / (n * n + 1)
                        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                    gap = self.min_uav_separation - distance + margin
                    positions[i] = np.clip(positions[i] - direction * gap / 2.0, lower, upper)
                    positions[j] = np.clip(positions[j] + direction * gap / 2.0, lower, upper)
                    moved = True
            if not moved:
                break
        return positions

    def _min_uav_distance(self) -> float:
        """Smallest 3D distance between two UAVs, inf with a single UAV"""
        positions = [uav.position for uav in self.uavs.values()]
        distances = [np.linalg.norm(positions[i] - positions[j])
                     for i in range(len(positions)) for j in range(i + 1, len(positions))]
        return float(min(distances)) if distances else float('inf')

    # ============================================
    # UEs DYNAMIC METHODS
    # ============================================

    def _spawn_new_hotspot(self):
        """Create a new hotspot at random location"""
        # Random position in service area
        x = np.random.uniform(self.service_area[0] * 0.1, self.service_area[0] * 0.9)
        y = np.random.uniform(self.service_area[1] * 0.1, self.service_area[1] * 0.9)
        
        # Random properties
        hotspot = {
            'position': np.array([x, y]),
            'radius': np.random.uniform(100.0, 500.0),  # Random size
            'max_strength': np.random.uniform(0.5, 1.0),  # Peak strength
            'current_strength': 0.0,  # Start at 0, will grow
            'lifetime': np.random.uniform(self.hotspot_min_lifetime, self.hotspot_max_lifetime),
            'age': 0.0,  # Current age in seconds
            'state': 'growing'  # States: 'growing', 'active', 'fading'
        }
        
        self.hotspots.append(hotspot)
        return hotspot

    def _update_hotspots(self):
        """Update hotspot lifecycles - called every T_L"""
        
        # 1. Update existing hotspots
        hotspots_to_remove = []
        
        for i, hotspot in enumerate(self.hotspots):
            hotspot['age'] += self.T_L
            
            # Lifecycle stages with smooth transitions
            growth_duration = hotspot['lifetime'] * 0.2  # 20% of life spent growing
            fade_start = hotspot['lifetime'] * 0.7  # Start fading at 70% of lifetime
            
            if hotspot['age'] < growth_duration:
                # Growing phase: strength increases from 0 to max_strength
                hotspot['state'] = 'growing'
                progress = hotspot['age'] / growth_duration
                hotspot['current_strength'] = hotspot['max_strength'] * progress
                
            elif hotspot['age'] < fade_start:
                # Active phase: maintain full strength
                hotspot['state'] = 'active'
                hotspot['current_strength'] = hotspot['max_strength']
                
            elif hotspot['age'] < hotspot['lifetime']:
                # Fading phase: strength decreases to 0
                hotspot['state'] = 'fading'
                fade_duration = hotspot['lifetime'] - fade_start
                fade_progress = (hotspot['age'] - fade_start) / fade_duration
                hotspot['current_strength'] = hotspot['max_strength'] * (1 - fade_progress)
                
            else:
                # Hotspot expired
                hotspots_to_remove.append(i)
        
        # 2. Remove expired hotspots
        for i in reversed(hotspots_to_remove):
            del self.hotspots[i]
        
        # 3. Randomly spawn new hotspots
        if len(self.hotspots) < self.hotspot_max_count:
            if np.random.random() < self.hotspot_spawn_rate:
                self._spawn_new_hotspot()
    
    def _get_hotspot_influence(self, ue_position):
        """Calculate attraction vector from all active hotspots"""
        total_attraction = np.zeros(2)
        
        for hotspot in self.hotspots:
            # Skip if hotspot is too weak
            if hotspot['current_strength'] < 0.1:
                continue
            
            # Calculate distance to this hotspot
            direction = hotspot['position'] - ue_position[:2]
            distance = np.linalg.norm(direction)
            
            # Only attract if within radius
            if distance < hotspot['radius'] and distance > 1.0:
                # Normalize direction
                direction = direction / distance
                
                # Attraction decreases with distance
                distance_factor = (1 - distance / hotspot['radius'])
                attraction_magnitude = hotspot['current_strength'] * distance_factor
                
                total_attraction += direction * attraction_magnitude
        
        return total_attraction

    def _update_ue_dynamics(self):
        """OPTIMIZED: Update UE positions, handle arrivals/departures"""
        
        # Reset statistics
        self.stats['arrivals'] = 0
        self.stats['departures'] = 0
        
        # Update hotspot lifecycles
        self._update_hotspots()
        
        # ============================================
        # OPTIMIZATION 1: Get active UEs as arrays
        # ============================================
        active_ues = [ue for ue in self.ues.values() if ue.is_active]
        
        if not active_ues:
            # No active UEs, just handle arrivals
            self._handle_arrivals()
            return
        
        num_active = len(active_ues)
        
        # ============================================
        # OPTIMIZATION 2: Vectorized hotspot influence
        # ============================================
        if self.hotspots:
            # Get all UE positions at once
            ue_positions = np.array([ue.position[:2] for ue in active_ues], dtype=np.float32)
            
            # Vectorized hotspot influence calculation
            hotspot_pulls = np.zeros((num_active, 2), dtype=np.float32)
            
            for hotspot in self.hotspots:
                if hotspot['current_strength'] < 0.1:
                    continue
                
                # Vectorized distance calculation for all UEs
                directions = hotspot['position'] - ue_positions  # (num_ues, 2)
                distances = np.linalg.norm(directions, axis=1)  # (num_ues,)
                
                # Mask for UEs within radius
                in_radius = (distances < hotspot['radius']) & (distances > 1.0)
                
                if not np.any(in_radius):
                    continue
                
                # Vectorized attraction calculation
                normalized_dirs = directions[in_radius] / distances[in_radius, np.newaxis]
                distance_factors = 1 - distances[in_radius] / hotspot['radius']
                attraction = (hotspot['current_strength'] * distance_factors)[:, np.newaxis] * normalized_dirs
                
                hotspot_pulls[in_radius] += attraction
        else:
            hotspot_pulls = np.zeros((num_active, 2), dtype=np.float32)
        
        # ============================================
        # OPTIMIZATION 3: Vectorized velocity updates
        # ============================================
        for i, ue in enumerate(active_ues):
            current_speed = np.linalg.norm(ue.velocity[:2])
            
            if current_speed > 0 and np.any(hotspot_pulls[i]) and self.city is None:
                # Blend velocity with hotspot pull
                ue.velocity[:2] = (
                    (1 - self.hotspot_attraction_strength) * ue.velocity[:2] +
                    self.hotspot_attraction_strength * hotspot_pulls[i] * current_speed
                )
        
        # ============================================
        # OPTIMIZATION 4: Vectorized position updates
        # ============================================
        for ue in active_ues:
            # Update position
            new_position = ue.position + ue.velocity * self.T_L
            
            # Boundary handling (vectorized per UE)
            for dim in range(2):
                if new_position[dim] < 0 or new_position[dim] > self.service_area[dim]:
                    ue.velocity[dim] = -ue.velocity[dim]
                    new_position[dim] = np.clip(new_position[dim], 0, self.service_area[dim])
            
            ue.position = new_position
        
        # ============================================
        # OPTIMIZATION 5: Batch random number generation
        # ============================================
        # Generate all random numbers at once (much faster)
        direction_changes = np.random.random(num_active) < self.change_direction_prob
        departures = np.random.random(num_active) < self.ue_departure_rate
        
        # Apply direction changes
        for i, ue in enumerate(active_ues):
            if direction_changes[i] and self.city is not None:
                speed = np.linalg.norm(ue.velocity[:2])
                if speed > 0:
                    # Turn into another street: a perpendicular one is entered at the nearest crossing
                    d = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0])][np.random.randint(4)]
                    if not any(np.allclose(d, a) for a in self._street_directions(ue.position[:2])):
                        ue.position[:2] = np.clip(np.round(ue.position[:2] / self.city.pitch) * self.city.pitch,
                                                  0.0, self.service_area)
                    ue.velocity[:2] = speed * d
            elif direction_changes[i]:
                speed = np.linalg.norm(ue.velocity[:2])
                if speed > 0:
                    new_direction = np.random.uniform(0, 2 * np.pi)
                    ue.velocity[0] = speed * np.cos(new_direction)
                    ue.velocity[1] = speed * np.sin(new_direction)
        
        # ============================================
        # OPTIMIZATION 6: Efficient departure handling
        # ============================================
        departed_ues = []
        for i, ue in enumerate(active_ues):
            if departures[i] and not getattr(ue, 'fixed', False):
                ue.is_active = False
                self.stats['departures'] += 1
                departed_ues.append(ue.id)
        
        # Batch cleanup
        for ue_id in departed_ues:
            if hasattr(self, 'ue_id_pool'):
                self.ue_id_pool.add(ue_id)
            del self.ues[ue_id]
        
        # ============================================
        # 7. Handle arrivals
        # ============================================
        self._handle_arrivals()

    def _update_ue_traffic_patterns(self):
        """
        This step's offered traffic of every active UE, as the mean over the step (the packet-level models of
        TR 38.812 are far finer than a 60 s step):
        - eMBB: active with probability activity_factor, then peak_rate (TS 22.261 Table 7.1-1, urban macro:
          50 Mbps experienced rate, activity factor 20 %); otherwise no demand this step.
        - URLLC (road-side units): rate continuously (TS 22.261 Table 7.2.3.2-1: 10 Mbps, small packets).
        - mMTC: periodic reports (TR 45.820 Annex E), a mean of mmtc_report_rate packets per second.
        A UE without demand gets no RBs and does not count in QoS or fairness
        """
        tp = self.traffic_patterns
        for ue in self.ues.values():
            if not ue.is_active:
                continue
            ue.packet_bytes = self.packet_bytes[ue.slice_type]
            if ue.slice_type == 'embb':
                ue.demand_bps = tp.embb.peak_rate if np.random.random() < tp.embb.activity_factor else 0.0
            elif ue.slice_type == 'urllc':
                ue.demand_bps = tp.urllc.rate
            else:
                ue.demand_bps = self.mmtc_report_rate * ue.packet_bytes * 8.0
            ue.packet_rate = ue.demand_bps / (ue.packet_bytes * 8.0)

    def _add_rsus(self):
        """
        URLLC users are road-side infrastructure of an ITS (TS 22.261 Sec. 7.2.3.2, wireless road-side
        infrastructure backhaul): stationary, "deployed alongside streets ... every 1-2 km". The streets are the
        TR 36.885 urban road grid (Annex A.1.2: blocks of 433 m x 250 m), centred in the area; RSUs sit every
        `spacing` m along it, antenna at 5 m (TR 36.885 A.1.4). With the city, each RSU moves to the nearest
        street crossing of the building grid
        """
        if self.rsu_params is None:
            return
        rp = self.rsu_params
        gx, gy = rp['block_size']
        width, height = self.service_area
        # Road segments as (start, direction, length): north-south roads at x = gx/2 + k gx, east-west at y = ...
        roads = [(np.array([x, 0.0]), np.array([0.0, 1.0]), height) for x in np.arange(gx / 2.0, width, gx)]
        roads += [(np.array([0.0, y]), np.array([1.0, 0.0]), width) for y in np.arange(gy / 2.0, height, gy)]
        total = sum(length for _, _, length in roads)
        spacing = rp['spacing']
        targets = np.arange(spacing / 2.0, total, spacing)  # arc-length positions along the joined roads
        start_of = np.cumsum([0.0] + [length for _, _, length in roads])
        for s in targets:
            k = np.searchsorted(start_of, s, side='right') - 1
            origin, direction, _ = roads[k]
            pos = np.array([*(origin + direction * (s - start_of[k])), rp['antenna_height']])
            if self.city is not None:
                pos[:2] = np.clip(np.round(pos[:2] / self.city.pitch) * self.city.pitch, 0.0, self.service_area)
            self._add_new_ue(slice_type='urllc', position=pos, fixed=True)

    def _handle_arrivals(self):
        """Separate function for handling arrivals (cleaner code)"""
        # RSUs are fixed infrastructure, outside the arriving and departing population
        active_ue_count = len([ue for ue in self.ues.values() if ue.is_active and not getattr(ue, 'fixed', False)])
        expected_arrivals = self.ue_arrival_rate * self.T_L
        num_arrivals = min(
            np.random.poisson(expected_arrivals),
            self.max_ues - active_ue_count
        )
        
        for _ in range(num_arrivals):
            self._add_new_ue()
            self.stats['arrivals'] += 1

    def _add_new_ue(self, slice_type: Optional[str] = None, position: Optional[np.ndarray] = None,
                    fixed: bool = False):
        """Add a new UE to the network - spawn near active hotspots WITH ID REUSE. A fixed UE (an RSU) is
        placed at `position`, never moves and never departs"""
        if fixed:
            if self.ue_id_pool:
                new_ue_id = self.ue_id_pool.pop()
            else:
                new_ue_id = self.next_ue_id
                self.next_ue_id += 1
            ue = UE(id=new_ue_id, position=np.asarray(position, dtype=np.float64), slice_type=slice_type,
                    velocity=np.zeros(3), is_active=True)
            ue.fixed = True
            self.ues[new_ue_id] = ue
            return
        
        # ============================================
        # Get UE ID (reuse from pool if available)
        # ============================================
        if self.ue_id_pool:
            # Reuse an ID from the pool
            new_ue_id = self.ue_id_pool.pop()
        else:
            # Generate new ID
            new_ue_id = self.next_ue_id
            self.next_ue_id += 1
        
        # ============================================
        # Position selection (existing logic)
        # ============================================
        # Find active hotspots (strength > 0.5)
        active_hotspots = [h for h in self.hotspots if h['current_strength'] > 0.5]
        
        # 70% chance to spawn near an active hotspot
        if active_hotspots and np.random.random() < 0.7:
            # Choose hotspot weighted by strength
            weights = [h['current_strength'] for h in active_hotspots]
            weights = np.array(weights) / sum(weights)
            hotspot = active_hotspots[np.random.choice(len(active_hotspots), p=weights)]
            
            # Spawn within hotspot radius
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, hotspot['radius'] * 0.6)
            
            x = hotspot['position'][0] + distance * np.cos(angle)
            y = hotspot['position'][1] + distance * np.sin(angle)
            
            # Clip to service area
            x = np.clip(x, 0, self.service_area[0])
            y = np.clip(y, 0, self.service_area[1])
        else:
            # Random position
            x = np.random.uniform(0, self.service_area[0])
            y = np.random.uniform(0, self.service_area[1])
        
        # ============================================
        # Velocity and slice type (existing logic)
        # ============================================
        speed = np.random.uniform(0, self.ue_max_initial_velocity)
        direction = np.random.uniform(0, 2 * np.pi)
        velocity_x = speed * np.cos(direction)
        velocity_y = speed * np.sin(direction)
        
        slice_type = np.random.choice(
            ["embb", "urllc", "mmtc"],
            p=[self.slice_probs["embb"], self.slice_probs["urllc"], self.slice_probs["mmtc"]]
        )
        
        # ============================================
        # Create UE with reused/new ID
        # ============================================
        z = 0.0
        if self.city is not None:
            # A UE in a footprint is indoor and stays put; outdoors it is a pedestrian walking along the street
            # at pedestrian_speed (TR 38.901 Table 7.2-1: 3 km/h). Height ue_height (Table 7.4.1-1: 1.5 m)
            z = self.ue_dynamics['ue_height']
            directions = self._street_directions(np.array([x, y]))
            if not directions:
                velocity_x = velocity_y = 0.0
            else:
                d = directions[np.random.randint(len(directions))]
                velocity_x, velocity_y = self.ue_dynamics['pedestrian_speed'] * d
        new_ue = UE(
            id=new_ue_id,  # Reused or new ID
            position=np.array([x, y, z]),
            slice_type=slice_type,
            velocity=np.array([velocity_x, velocity_y, 0.0]),
            is_active=True
        )
        
        self.ues[new_ue_id] = new_ue

    def _street_directions(self, xy: np.ndarray) -> List[np.ndarray]:
        """Unit directions a pedestrian at xy can walk: along a north-south street +-y, along an east-west street
        +-x, all four at a crossing, none inside a footprint"""
        local = xy - np.floor(xy / self.city.pitch) * self.city.pitch
        in_street = (local < self.city.offset) | (local >= self.city.offset + self.city.width)
        directions = []
        if in_street[0]:  # x lies in a north-south street
            directions += [np.array([0.0, 1.0]), np.array([0.0, -1.0])]
        if in_street[1]:
            directions += [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
        return directions

    def _update_uavs_queuing_stats(self):
        '''Calculate the average queuing delay that UEs experience at their associated UAVs at each T_L period'''
        for uav in self.uavs.values():

            # Average packet size (bytes)
            served = [ue for ue in self.ues.values()
                      if ue.assigned_uav == uav.id and ue.is_active and getattr(ue, 'packet_rate', 0.0) > 0]
            packet_sizes = [ue.packet_bytes for ue in served]
            avg_packet_size = np.mean(packet_sizes) if packet_sizes else 0.0

            
            # Cummulative arrival rate (packets/sec)
            arrival_rates = [ue.packet_rate for ue in served]
            uav_arrival_rate = np.sum(arrival_rates) if arrival_rates else 0.0
            
            # Cummulative service rate (packets/sec)
            througputs_of_ues_in_uav = [self._get_cached_throughput(ue) for ue in self.ues.values() if ue.assigned_uav == uav.id and ue.is_active]  # in bps
            uav_throughput_bps = np.sum(througputs_of_ues_in_uav) if througputs_of_ues_in_uav else 0.0
            if self.backhaul is not None:
                # Packets leave the UAV over the backhaul, so it sets the service rate as well
                uav_throughput_bps = min(uav_throughput_bps, self.backhaul_capacity[uav.id])
            packet_bits = 1e3 + avg_packet_size * 8
            if self.backhaul is not None and uav_arrival_rate > 0:
                # Mean over packets, not UEs, so that rho is offered bits over service bits and follows the
                # backhaul load. Only with a backhaul: without one, results must stay comparable with earlier runs
                packet_bits = 8.0 * np.dot(packet_sizes, arrival_rates) / uav_arrival_rate
            uav_service_rate = uav_throughput_bps / packet_bits  # packets per second

            # Buffer size in packets
            uav_buffer_size = uav.buffer_size / packet_bits # in packets

            # CALCULATE QUEUING DELAY
            rho = uav_arrival_rate / max(uav_service_rate, 1)  # Utilization
            if rho >= 1:
                uav.avg_queuing_delay_ms = 99
            elif uav_service_rate == 0:
                uav.avg_queuing_delay_ms = 999.0
            else:
                uav.avg_queuing_delay_ms = rho / (2 * uav_service_rate * (1 - rho)) * 1000  # in ms, M/D/1 formula
                if self.backhaul is not None:
                    # The overload value caps it, so delay doesn't fall as load passes rho = 1.
                    # Only with a backhaul: without one, results must stay comparable with earlier runs
                    uav.avg_queuing_delay_ms = min(uav.avg_queuing_delay_ms, 99.0)

            # CALCULATE DROP RATE (M/M/1/K blocking probability)
            if uav_service_rate == 0:
                uav.avg_drop_rate = 1.0 if uav_arrival_rate > 0 else 0.0
            elif rho > 1 and (self.backhaul is None or (uav_buffer_size + 1) * np.log(rho) > 700.0):
                # Limit for an overloaded queue; the finite-K formula overflows here.
                # With a backhaul the formula runs up to that point, so the drop rate is continuous at rho = 1
                uav.avg_drop_rate = 1.0 - 1.0 / rho
            elif rho == 1:
                uav.avg_drop_rate = 1 / (uav_buffer_size + 1)
            else:
                # The textbook numerator is rho^K. The one rho too many is harmless below rho = 1,
                # and stays without a backhaul so that results remain comparable with earlier runs
                numerator_power = uav_buffer_size + 1 if self.backhaul is None else uav_buffer_size
                uav.avg_drop_rate = ( (rho**numerator_power) * (1 - rho) ) / (1 - rho**(uav_buffer_size + 1))
            uav.avg_drop_rate = np.clip(uav.avg_drop_rate, 0.0, 1.0)

            # print(f"UAV {uav.id} - Arrival rate: {uav_arrival_rate:<10.2f} pkt/s | Service rate: {uav_service_rate:<12.2f} pkt/s | Rho: {rho:<10.4f} | Buffer size: {uav_buffer_size:<5.0f} pkts | Avg Delay: {uav.avg_queuing_delay_ms:<10.6f} ms | Drop Rate: {uav.avg_drop_rate:<10.4f}")
    # ============================================
    # ASSOCIATION METHODS
    # ============================================

    def _associate_ues_to_uavs(self):
        """Associate UEs to UAVs, by reference SNR with the 3GPP antenna, else to the nearest UAV in beam"""
        self.stats['handovers'] = 0
        self.stats['uncovered_ues'] = 0
        self.stats['covered_ues'] = 0
        
        # Store previous assignments
        previous_assignments = {ue.id: ue.assigned_uav for ue in self.ues.values() if ue.is_active}
        
        # Get all active UEs
        active_ues = [ue for ue in self.ues.values() if ue.is_active]
        if not active_ues:
            return
        
        if self.antenna is not None:
            self._associate_by_reference_snr(active_ues)
        else:
            self._associate_by_beam(active_ues)

        # Handover tracking (existing code)
        for ue in active_ues:
            old_uav = previous_assignments.get(ue.id)
            new_uav = ue.assigned_uav
            if new_uav is None:
                self.stats['uncovered_ues'] += 1
            else:
                self.stats['covered_ues'] += 1

            handover_info = self.handover_tracker.check_handover(
                ue.id, old_uav, new_uav, self.current_time
            )

            if not hasattr(ue, 'handover_penalty_ms'):
                ue.handover_penalty_ms = 0.0

            if handover_info['handover']:
                ue.handover_penalty_ms = handover_info['delay_penalty_ms']
                self.stats['handovers'] += 1
            else:
                ue.handover_penalty_ms = handover_info['delay_penalty_ms']

    def _associate_by_reference_snr(self, active_ues: List[UE]):
        """Strongest reference signal, covered if its SNR per RB at the UAV's power reaches the minimum"""
        gains_db = self._link_gains_db(active_ues)  # (num_ues, num_uavs)
        ref_power_dbw = 10.0 * np.log10([uav.current_power / self.total_rbs for uav in self.uavs.values()])
        ref_snr_db = gains_db + ref_power_dbw[np.newaxis, :] - 10.0 * np.log10(self.noise_power)
        best = np.argmax(ref_snr_db, axis=1)
        uav_ids_list = list(self.uavs.keys())
        min_snr_db = self.antenna['min_association_snr_db']
        # Event A3 (TS 38.331): hand over only when the best UAV beats the serving one by the hysteresis
        hysteresis_db = self.antenna['handover_hysteresis_db'] if 'handover_hysteresis_db' in self.antenna else 0.0
        for i, ue in enumerate(active_ues):
            target = best[i]
            if ue.assigned_uav in uav_ids_list:
                serving = uav_ids_list.index(ue.assigned_uav)
                if (ref_snr_db[i, serving] >= min_snr_db
                        and ref_snr_db[i, target] - ref_snr_db[i, serving] <= hysteresis_db):
                    target = serving
            ue.assigned_uav = uav_ids_list[target] if ref_snr_db[i, target] >= min_snr_db else None

    def _associate_by_beam(self, active_ues: List[UE]):
        """Nearest UAV whose beam cone contains the UE"""
        # Prepare arrays
        ue_positions = np.array([ue.position for ue in active_ues], dtype=np.float32)
        uav_positions = np.array([uav.position for uav in self.uavs.values()], dtype=np.float32)
        uav_beam_dirs = np.array([uav.beam_direction for uav in self.uavs.values()], dtype=np.float32)
        uav_beam_angles = np.array([uav.beam_angle for uav in self.uavs.values()], dtype=np.float32)
        
        # Compute ALL distances at once (num_ues × num_uavs matrix)
        # Shape: (num_ues, 1, 3) - (1, num_uavs, 3) = (num_ues, num_uavs, 3)
        diff_vectors = ue_positions[:, np.newaxis, :] - uav_positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff_vectors, axis=2)  # (num_ues, num_uavs)
        
        # Compute ALL beam angles at once
        # Normalize direction vectors
        distances_safe = np.maximum(distances, 1e-6)
        normalized_dirs = diff_vectors / distances_safe[:, :, np.newaxis]
        
        # Dot product for all UE-UAV pairs
        # (num_ues, num_uavs, 3) · (1, num_uavs, 3) -> (num_ues, num_uavs)
        dot_products = np.sum(normalized_dirs * uav_beam_dirs[np.newaxis, :, :], axis=2)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        
        # Calculate angles in degrees
        angles = np.degrees(np.arccos(dot_products))  # (num_ues, num_uavs)
        
        # Check beam constraints
        within_beam = angles <= uav_beam_angles[np.newaxis, :]  # (num_ues, num_uavs)
        
        # Set invalid distances to infinity
        distances_masked = distances.copy()
        distances_masked[~within_beam] = np.inf
        
        # Find best UAV for each UE
        best_uav_indices = np.argmin(distances_masked, axis=1)
        
        # Assign UEs to UAVs
        uav_ids_list = list(self.uavs.keys())
        for i, ue in enumerate(active_ues):
            min_dist = distances_masked[i, best_uav_indices[i]]
            if min_dist < np.inf:
                ue.assigned_uav = uav_ids_list[best_uav_indices[i]]
            else:
                ue.assigned_uav = None

    def _form_demand_areas(self):
        """OPTIMIZED: Cache UE-to-DA mapping to avoid repeated lookups"""
        self.demand_areas.clear()
        self.da_counter = 0

        # Pre-compute distance thresholds
        near_max = self.da_distance_thresholds['near_max']
        medium_max = self.da_distance_thresholds['medium_max']
        
        # Build UE-to-DA mapping dictionary for fast lookup
        self.ue_to_da_map = {}  # NEW: Cache this!
        
        for uav_id in range(self.num_uavs):
            uav_pos = self.uavs[uav_id].position
            
            # Get active UEs for this UAV (vectorize distance calc)
            uav_ues = [ue for ue in self.ues.values() 
                    if ue.assigned_uav == uav_id and ue.is_active]
            
            if not uav_ues:
                # Still create empty DAs
                for slice_type in ["embb", "urllc", "mmtc"]:
                    for level_name in ['Near', 'Medium', 'Far']:
                        da = DemandArea(
                            id=self.da_counter,
                            uav_id=uav_id,
                            slice_type=slice_type,
                            user_ids=[],
                            distance_level=level_name
                        )
                        da.center_position = uav_pos.copy()
                        self.demand_areas[self.da_counter] = da
                        self.da_counter += 1
                continue
            
            # Vectorized distance calculation for all UEs
            ue_positions = np.array([ue.position for ue in uav_ues])
            distances = np.linalg.norm(ue_positions - uav_pos, axis=1)
            
            # Group by slice type
            for slice_type in ["embb", "urllc", "mmtc"]:
                # Boolean mask for this slice
                slice_mask = np.array([ue.slice_type == slice_type for ue in uav_ues])
                slice_ues = [ue for ue, m in zip(uav_ues, slice_mask) if m]
                slice_distances = distances[slice_mask]
                
                # Categorize by distance (vectorized)
                distance_groups = {
                    'Near': [ue for ue, d in zip(slice_ues, slice_distances) if d <= near_max],
                    'Medium': [ue for ue, d in zip(slice_ues, slice_distances) if near_max < d <= medium_max],
                    'Far': [ue for ue, d in zip(slice_ues, slice_distances) if d > medium_max]
                }
                
                # Create DAs
                for level_name in ['Near', 'Medium', 'Far']:
                    ue_group = distance_groups[level_name]
                    
                    da = DemandArea(
                        id=self.da_counter,
                        uav_id=uav_id,
                        slice_type=slice_type,
                        user_ids=[ue.id for ue in ue_group],
                        distance_level=level_name
                    )
                    
                    if ue_group:
                        positions = np.array([ue.position for ue in ue_group])
                        da.center_position = np.mean(positions, axis=0)
                        
                        # Update mapping
                        for ue in ue_group:
                            ue.assigned_da = da.id
                            self.ue_to_da_map[ue.id] = da.id
                    else:
                        da.center_position = uav_pos.copy()
                    
                    self.demand_areas[self.da_counter] = da
                    self.da_counter += 1

    # ============================================
    # BEAM CALCULATION METHODS
    # ============================================

    def _calculate_beam_angle(self, uav: UAV, ue: UE) -> float:
        """
        Calculate the angle between UAV's beam direction and the direction to UE.
        Returns angle in degrees.
        
        Args:
            uav: UAV object
            ue: UE object
            
        Returns:
            Angle in degrees from UAV's beam center to UE
        """
        # Vector from UAV to UE
        vec_to_ue = ue.position - uav.position
        distance = np.linalg.norm(vec_to_ue)
        
        if distance < 1e-6:  # Avoid division by zero
            return 0.0
        
        # Normalize
        vec_to_ue_norm = vec_to_ue / distance
        
        # Calculate angle using dot product
        # angle = arccos(a · b) where a and b are unit vectors
        dot_product = np.dot(uav.beam_direction, vec_to_ue_norm)
        
        # Clamp to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
  
    def _is_ue_in_beam(self, uav: UAV, ue: UE) -> bool:
        """
        Check if UE is within UAV's beam coverage.
        
        Args:
            uav: UAV object
            ue: UE object
            
        Returns:
            True if UE is within beam angle, False otherwise
        """
        angle = self._calculate_beam_angle(uav, ue)
        return angle <= uav.beam_angle

    def get_uav_beam_info(self, uav_id: int) -> Dict:
        """
        Get beam information for visualization.
        
        Args:
            uav_id: UAV ID
            
        Returns:
            Dictionary with beam parameters
        """
        if uav_id not in self.uavs:
            return None
        
        uav = self.uavs[uav_id]
        
        # Calculate beam radius at ground level
        height = uav.position[2]
        if height > 0:
            beam_radius = height * np.tan(np.radians(uav.beam_angle))
        else:
            beam_radius = 0
        
        return {
            'position': uav.position.copy(),
            'beam_angle': uav.beam_angle,
            'beam_direction': uav.beam_direction.copy(),
            'beam_radius_ground': beam_radius,
            'height': height
        }
    
    def get_ue_beam_status(self, ue_id: int) -> Dict:
        """
        Check if UE is within its assigned UAV's beam.
        
        Args:
            ue_id: UE ID
            
        Returns:
            Dictionary with coverage status
        """
        if ue_id not in self.ues:
            return None
        
        ue = self.ues[ue_id]
        if not ue.is_active or ue.assigned_uav is None:
            return {'covered': False, 'angle': None}
        
        uav = self.uavs[ue.assigned_uav]
        angle = self._calculate_beam_angle(uav, ue)
        covered = self._is_ue_in_beam(uav, ue)
        
        return {
            'covered': covered,
            'angle': angle,
            'max_angle': uav.beam_angle,
            'margin': uav.beam_angle - angle  # Positive = inside, negative = outside
        }

    # ============================================
    # CACHING METHODS
    # ============================================

    def _update_sinr_cache(self):
        if self.sinr_cache_valid:
            return
        
        self.sinr_cache.clear()
        
        # Collect all UE-RB pairs
        calculation_list = []
        ue_ids, serving_uav_ids = [], []
        
        for ue in self.ues.values():
            if not ue.is_active or ue.assigned_uav is None or not ue.assigned_rb:
                continue
            calculation_list.append(ue)
            ue_ids.append(ue.id)
            serving_uav_ids.append(ue.assigned_uav)
        
        if not calculation_list:
            self.sinr_cache_valid = True
            return
        
        # Prepare arrays for Numba
        ue_positions = np.array([ue.position for ue in calculation_list], dtype=np.float32)
        uav_positions = np.array([uav.position for uav in self.uavs.values()], dtype=np.float32)
        # Each UAV splits its transmit power equally across the RBs it actually uses
        rbs_in_use = [sum(rb.allocated_ue_id != -1 for rb in uav.RBs) for uav in self.uavs.values()]
        uav_powers = np.array([
            uav.current_power / max(1, n_rbs)
            for uav, n_rbs in zip(self.uavs.values(), rbs_in_use)
        ], dtype=np.float32)
        # Expected power on another UAV's RB: (P / n_rbs) per RB, on n_rbs / total_rbs of the band.
        # This is P / total_rbs however many RBs the UAV uses, and nothing if it transmits on none
        interference_powers = np.array([
            uav.current_power / self.total_rbs if n_rbs > 0 else 0.0
            for uav, n_rbs in zip(self.uavs.values(), rbs_in_use)
        ], dtype=np.float32)

        # Map serving UAV IDs to indices
        uav_ids_list = list(self.uavs.keys())
        uav_index_map = {uid: idx for idx, uid in enumerate(uav_ids_list)}
        serving_indices = np.array([uav_index_map[uid] for uid in serving_uav_ids], dtype=np.int32)

        
        # Call Numba-accelerated function
        # sinr_db = calculate_sinr_batch_numba(
        #     ue_positions, uav_positions, uav_powers,
        #     serving_indices, self.noise_power, self.path_loss_exponent
        # )
        if self.antenna is not None:
            sinr_db = self._sinr_db(calculation_list, uav_powers, serving_indices, interference_powers)
        else:
            sinr_db = calculate_sinr_batch_numba_a2g(ue_positions, uav_positions, uav_powers,
                serving_indices, self.noise_power, self.carrier_frequency,
                interference_powers, self.sidelobe_gain)
        
        # Fill cache
        for i, (ue_id, uav_id) in enumerate(zip(ue_ids, serving_uav_ids)):
            self.sinr_cache[(ue_id, uav_id)] = sinr_db[i]
        
        self.sinr_cache_valid = True

    def _link_gains_db(self, ues: List[UE]) -> np.ndarray:
        """
        Gain in dB of every UE-UAV link: UAV antenna gain minus path loss, shape (num_ues, num_uavs).
        Path loss is Al-Hourani's A2G model (FSPL plus eta). With shadowing, the LoS state is drawn
        from P_LoS and a log-normal term of TR 36.777 UMa-AV is added; without it, eta is its expectation.
        """
        ue_positions = np.array([ue.position for ue in ues], dtype=np.float64)
        uav_positions = np.array([uav.position for uav in self.uavs.values()], dtype=np.float64)
        diff = uav_positions[np.newaxis, :, :] - ue_positions[:, np.newaxis, :]
        distance_2d = np.maximum(np.hypot(diff[..., 0], diff[..., 1]), 1e-6)
        distance_3d = np.maximum(np.linalg.norm(diff, axis=2), 1.0)
        elevation_deg = np.degrees(np.arctan(uav_positions[np.newaxis, :, 2] / distance_2d))

        a2g = self.a2g
        p_los = 1.0 / (1.0 + a2g['a'] * np.exp(np.clip(-a2g['b'] * (elevation_deg - a2g['a']), -20.0, 20.0)))
        fspl_db = 20.0 * np.log10(distance_3d) + 20.0 * np.log10(self.carrier_frequency / 1e6) - 27.55
        if self.city is not None:
            # Geometric LoS state, correlated shadowing and O2I loss from _update_channel_state
            los = np.array([ue.los for ue in ues])
            shadow_db = np.array([ue.shadow_db for ue in ues])
            o2i_db = np.array([ue.o2i_db for ue in ues])[:, np.newaxis]
            path_loss_db = fspl_db + np.where(los, a2g['eta_los'], a2g['eta_nlos']) + shadow_db + o2i_db
        elif self.shadowing is None:
            path_loss_db = fspl_db + a2g['eta_los'] * p_los + a2g['eta_nlos'] * (1.0 - p_los)
        else:
            draws = [self._channel_draw(ue.id) for ue in ues]
            uniform = np.array([d[0] for d in draws])
            normal = np.array([d[1] for d in draws])
            los = uniform < p_los
            heights = uav_positions[np.newaxis, :, 2]
            sigma_los = self.shadowing['sigma_los_scale'] * np.exp(-self.shadowing['sigma_los_decay'] * heights)
            sigma = np.where(los, sigma_los, self.shadowing['sigma_nlos'])
            path_loss_db = fspl_db + np.where(los, a2g['eta_los'], a2g['eta_nlos']) + sigma * normal

        if self.antenna is None:
            return -path_loss_db
        # Element pattern of TR 38.901 Table 7.3-1 around the downward boresight
        off_axis_deg = np.degrees(np.arccos(np.clip(diff[..., 2] / distance_3d, -1.0, 1.0)))
        attenuation_db = np.minimum(12.0 * (off_axis_deg / self.antenna['beamwidth_3db']) ** 2,
                                    self.antenna['max_attenuation_db'])
        return self.antenna['max_gain_dbi'] - attenuation_db - path_loss_db

    def _update_channel_state(self):
        """
        Per active UE and UAV, once a step, with the explicit city:
        - LoS: the UE-UAV segment clears every building (City.line_of_sight).
        - Indoor: the UE stands in a footprint. O2I loss of TR 38.901 Table 7.4.3-3 (single frequency < 6 GHz):
          wall 20 dB plus 0.5 dB/m over d_2D-in ~ U(0, 25 m), drawn once per UE, with no extra spread.
        - Shadowing: N(0, sigma^2) with sigma by state (TR 38.901 Table 7.4.1-1 UMa: LoS 4, NLoS 6 dB; Table
          7.4.3-3: 7 dB indoor), exponentially correlated (ITU-R P.1816, TR 38.901 7.4.4) over the distance the
          UE and the UAV moved since the last step, with the decorrelation distances of Table 7.5-6 (UMa).
        - Rician K-factor of the LoS link, N(mean, std) dB (TR 38.901 Table 7.5-6 UMa), drawn once per UE.
        """
        if self.city is None:
            return
        active = [ue for ue in self.ues.values() if ue.is_active]
        uav_positions = np.array([uav.position for uav in self.uavs.values()], dtype=np.float64)
        n_uavs = len(uav_positions)
        if self._sf_uav_positions is None:
            self._sf_uav_positions = uav_positions.copy()
        uav_moved = np.linalg.norm(uav_positions - self._sf_uav_positions, axis=1)
        self._sf_uav_positions = uav_positions.copy()
        if not active:
            return

        ue_positions = np.array([ue.position for ue in active], dtype=np.float64)
        indoor = self.city.is_indoor(ue_positions[:, :2])
        starts = np.repeat(ue_positions, n_uavs, axis=0)
        ends = np.tile(uav_positions, (len(active), 1))
        los = self.city.line_of_sight(starts, ends).reshape(len(active), n_uavs)

        sh, o2i, ff = self.shadowing, self.o2i, self.fast_fading
        for i, ue in enumerate(active):
            if not hasattr(ue, 'shadow_z'):
                ue.shadow_z = np.random.standard_normal(n_uavs)
                ue.shadow_anchor = ue.position.copy()
                ue.depth_in = np.random.uniform(0.0, o2i['max_indoor_distance'])
                ue.k_factor_db = np.random.normal(ff['k_factor_mean_db'], ff['k_factor_std_db'], n_uavs) \
                    if ff is not None else np.zeros(n_uavs)
            else:
                moved = np.linalg.norm(ue.position - ue.shadow_anchor) + uav_moved
                ue.shadow_anchor = ue.position.copy()
                if indoor[i]:
                    d_cor = np.full(n_uavs, sh['decorrelation_o2i'])
                else:
                    d_cor = np.where(los[i], sh['decorrelation_los'], sh['decorrelation_nlos'])
                rho = np.exp(-moved / d_cor)
                ue.shadow_z = rho * ue.shadow_z + np.sqrt(1.0 - rho ** 2) * np.random.standard_normal(n_uavs)
            if indoor[i]:
                sigma = np.full(n_uavs, sh['sigma_o2i'])
            else:
                sigma = np.where(los[i], sh['sigma_los'], sh['sigma_nlos'])
            ue.los = los[i]
            ue.indoor = bool(indoor[i])
            ue.shadow_db = sigma * ue.shadow_z
            ue.o2i_db = o2i['wall_loss_db'] + o2i['indoor_loss_per_m'] * ue.depth_in if indoor[i] else 0.0

    def _build_fading_table(self) -> Dict:
        """
        Mean spectral efficiency over fast fading, E[min(att * log2(1 + g * SINR), se_max)], tabulated over
        the mean SINR and the Rician K-factor (unit-mean power gain g). Row K = -inf is Rayleigh. A step lasts
        far longer than the coherence time, so the step's rate is this ergodic mean. Fixed-seed samples, so
        the table is the same in every run and does not touch the global RNG
        """
        rng = np.random.default_rng(20261105)
        n = 20000
        scatter = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
        sinr_db = np.arange(-10.0, 50.0 + 1e-9, 0.25)
        k_db = np.arange(-10.0, 40.0 + 1e-9, 1.0)
        sinr = 10.0 ** (sinr_db / 10.0)

        def mean_se(gain):
            se = self._se_from_sinr(sinr[:, np.newaxis] * gain[np.newaxis, :])
            return se.mean(axis=1)

        rician = np.empty((len(k_db), len(sinr_db)))
        for j, k in enumerate(10.0 ** (k_db / 10.0)):
            gain = np.abs(np.sqrt(k / (k + 1.0)) + np.sqrt(1.0 / (k + 1.0)) * scatter) ** 2
            rician[j] = mean_se(gain)
        return {'sinr_db': sinr_db, 'k_db': k_db, 'rician': rician, 'rayleigh': mean_se(np.abs(scatter) ** 2)}

    def _se_from_sinr(self, sinr_linear):
        """Attenuated Shannon with SNR_eff, capped at the top CQI, and 0 below the lowest CQI"""
        se = np.minimum(self.shannon_attenuation * np.log2(1.0 + sinr_linear / self.snr_efficiency),
                        self.max_spectral_efficiency)
        return np.where(se >= self.min_spectral_efficiency, se, 0.0)

    def _spectral_efficiency(self, sinr_db: float, ue: UE) -> float:
        """Attenuated Shannon, capped; with fast fading, its mean over Rician (LoS outdoor) or Rayleigh fading"""
        if self._se_table is None or self.city is None:
            return float(self._se_from_sinr(10.0 ** (sinr_db * 0.1)))
        table = self._se_table
        j = list(self.uavs.keys()).index(ue.assigned_uav)
        if ue.los[j] and not ue.indoor:
            k_db = table['k_db']
            pos = (np.clip(ue.k_factor_db[j], k_db[0], k_db[-1]) - k_db[0]) / (k_db[1] - k_db[0])
            lo = min(int(pos), len(k_db) - 2)
            w = pos - lo
            se_lo = np.interp(sinr_db, table['sinr_db'], table['rician'][lo])
            se_hi = np.interp(sinr_db, table['sinr_db'], table['rician'][lo + 1])
            return float((1.0 - w) * se_lo + w * se_hi)
        return float(np.interp(sinr_db, table['sinr_db'], table['rayleigh']))

    def _channel_draw(self, ue_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """This step's random numbers for the UE's links, drawn on first use"""
        if ue_id not in self.channel_draws:
            self.channel_draws[ue_id] = (np.random.uniform(size=self.num_uavs),
                                         np.random.standard_normal(self.num_uavs))
        return self.channel_draws[ue_id]

    def _sinr_db(self, ues: List[UE], uav_powers: np.ndarray, serving_indices: np.ndarray,
                 interference_powers: np.ndarray) -> np.ndarray:
        """SINR per RB with the link gains of _link_gains_db, clipped like the numba path"""
        gains = 10.0 ** (self._link_gains_db(ues) / 10.0)  # (num_ues, num_uavs)
        rows = np.arange(len(ues))
        signal = gains[rows, serving_indices] * uav_powers[serving_indices]
        received = gains * interference_powers[np.newaxis, :]
        interference = received.sum(axis=1) - received[rows, serving_indices]
        sinr_db = 10.0 * np.log10(signal / (interference + self.noise_power) + 1e-10)
        return np.clip(sinr_db, -10.0, 50.0).astype(np.float32)

    def _update_delay_cache(self):
        """
        Calculate ALL delays at once
        """
        self.delay_cache.clear()
        
        # Collect all active UEs with RBs
        active_ues = [
            (ue_id, ue) for ue_id, ue in self.ues.items()
            if ue.is_active and ue.assigned_rb and ue.assigned_uav is not None
        ]
        
        if not active_ues:
            self.delay_cache_valid = True
            return
        # ========================
        # Propagation delays
        # ========================
        ue_positions = np.array([ue.position for _, ue in active_ues], dtype=np.float32)
        uav_positions = np.array([
            self.uavs[ue.assigned_uav].position for _, ue in active_ues
        ], dtype=np.float32)
        
        distances = np.linalg.norm(ue_positions - uav_positions, axis=1)
        t_prop = (distances / 3e8) * 1000  # ms

        # ========================
        # Transmission delays
        # ========================

        packet_size_bits = np.array([ue.packet_bytes * 8 for _, ue in active_ues], dtype=np.float32)
        throughputs = np.array([np.clip(self._get_cached_throughput(ue), 0.1, None) for _, ue in active_ues], dtype=np.float32)
        t_trans = packet_size_bits / throughputs * 1000  # ms

        # ========================
        # Retransmission delays
        # ========================

        per_list = np.array([self._calculate_packet_error_rate(ue) for _, ue in active_ues], dtype=np.float32)
        t_retx = per_list * 4 * 8.0  # ms

        # ========================
        # Queuing delays
        # ========================

        t_queue = np.array([self.uavs[ue.assigned_uav].avg_queuing_delay_ms for _, ue in active_ues], dtype=np.float32)

        # ========================
        # Scheduling delays
        # ========================

        t_sched = np.zeros(len(active_ues), dtype=np.float32)

        # ========================
        # Handover delays
        # ========================

        # A handover interrupts D ms of the T_L: that share of packets waits D/2 on average, so the
        # mean added delay is D^2 / (2 T_L). The full D would apply to every packet of the step
        t_handover = np.array([getattr(ue, 'handover_penalty_ms', 0.0) ** 2 / (2.0 * self.T_L * 1000.0)
                               for _, ue in active_ues], dtype=np.float32)

        # ========================
        # Processing delays
        # ========================

        t_proc = np.full(len(active_ues), self.processing_delay_ms, dtype=np.float32)

        # Process each UE's delay components
        for i, (ue_id, ue) in enumerate(active_ues):
            
            # Total delay
            total_delay = (
                t_prop[i] + t_trans[i] + t_retx[i] +
                t_queue[i] + t_sched[i] +
                t_handover[i] +
                t_proc[i]
            )
            
            # Cache result
            self.delay_cache[ue_id] = {
                'total': total_delay,
                'breakdown': {
                    'propagation': t_prop[i],
                    'transmission': t_trans[i],
                    'retransmission': t_retx[i],
                    'queuing': t_queue[i],
                    'scheduling': t_sched[i],
                    'handover': t_handover[i],
                    'processing': t_proc[i],

                }
            }
        
        self.delay_cache_valid = True

    def _update_throughput_cache(self):
        
        self.throughput_cache.clear()
        
        for ue in self.ues.values():
            if not ue.is_active or ue.assigned_uav is None or len(ue.assigned_rb) == 0:
                self.throughput_cache[ue.id] = 0.0
                continue
            # Calculate total throughput across all RBs
            ue_throughput = 0.0

            sinr_db = self._get_cached_sinr(ue, self.uavs[ue.assigned_uav])
            spectral_efficiency = self._spectral_efficiency(sinr_db, ue)
            ue_throughput += self.rb_bandwidth * spectral_efficiency * len(ue.assigned_rb)  # in bps
            
            self.throughput_cache[ue.id] = ue_throughput

        if self.backhaul is not None:
            self._apply_backhaul_limit()

        self.throughput_cache_valid = True

    def _backhaul_capacity(self, uav: UAV) -> float:
        """Capacity (bps) of the UAV's own band to the gateway"""
        delta = uav.position - self.backhaul_gateway
        distance_2d = max(np.hypot(delta[0], delta[1]), 1e-6)
        distance_3d = max(np.linalg.norm(delta), 1.0)
        if getattr(self.backhaul, 'model', None) == 'mmwave':
            p_los = None
            if self.city is not None:
                # Geometric LoS between the gateway (on a street at the area's edge) and the UAV
                p_los = float(self.city.line_of_sight(self.backhaul_gateway[np.newaxis, :],
                                                      uav.position[np.newaxis, :])[0])
            return self._mmwave_backhaul_capacity(distance_2d, distance_3d, uav.position[2], p_los)

        # The access link's A2G and rate models
        elevation_deg = np.degrees(np.arctan(abs(delta[2]) / distance_2d))
        a2g = self.a2g
        p_los = 1.0 / (1.0 + a2g['a'] * np.exp(np.clip(-a2g['b'] * (elevation_deg - a2g['a']), -20.0, 20.0)))
        fspl_db = 20.0 * np.log10(distance_3d) + 20.0 * np.log10(self.carrier_frequency / 1e6) - 27.55
        path_loss_db = fspl_db + a2g['eta_los'] * p_los + a2g['eta_nlos'] * (1.0 - p_los)

        gain_db = self.backhaul.antenna_gain_db - path_loss_db
        snr = self.backhaul.tx_power * 10.0 ** (gain_db / 10.0) / self.backhaul_noise_power
        spectral_efficiency = min(self.shannon_attenuation * np.log2(1.0 + snr), self.max_spectral_efficiency)
        return self.backhaul.bandwidth * spectral_efficiency

    def _mmwave_backhaul_capacity(self, distance_2d: float, distance_3d: float, uav_height: float,
                                  p_los: Optional[float] = None) -> float:
        """
        mmWave backhaul of Galkin, Kibilda & DaSilva (arXiv:1710.10807): path loss c * d^-alpha with
        alpha by LoS state, LoS probability of their Eq. 3 (ITU-R P.1410 buildings, both ends elevated).
        The capacity is the mean over the LoS state of the attenuated Shannon rate. No fading, no interference.
        """
        bh = self.backhaul
        h_gw = self.backhaul_gateway[2]
        # Eq. 3: product over the buildings crossed, d_i = floor(r * sqrt(beta * delta)), r in km
        n_buildings = int(np.floor(distance_2d / 1000.0 * np.sqrt(bh.buildings_per_km2 * bh.built_up_ratio)))
        n = np.arange(max(0, n_buildings - 1) + 1) if n_buildings > 0 else np.arange(0)
        clearance = max(uav_height, h_gw) - (n + 0.5) * abs(uav_height - h_gw) / max(n_buildings, 1)
        if p_los is None:
            p_los = float(np.prod(1.0 - np.exp(-clearance ** 2 / (2.0 * bh.building_height_scale ** 2))))

        capacity = 0.0
        for prob, alpha in ((p_los, bh.path_loss_exponent_los), (1.0 - p_los, bh.path_loss_exponent_nlos)):
            path_loss_db = bh.path_loss_constant_db + 10.0 * alpha * np.log10(distance_3d)
            snr = bh.tx_power * 10.0 ** ((bh.antenna_gain_db - path_loss_db) / 10.0) / bh.noise_power
            spectral_efficiency = min(self.shannon_attenuation * np.log2(1.0 + snr), self.max_spectral_efficiency)
            capacity += prob * bh.bandwidth * spectral_efficiency
        return capacity

    def _apply_backhaul_limit(self):
        """
        The backhaul carries what the UEs send, min(access rate, offered load), not the spare access capacity.
        If that exceeds the capacity, all the UAV's UEs slow down by the same factor. This is continuous at
        load 1 and leaves the shares within a UAV as they were. The queue is limited by the capacity at
        any load, see _update_uavs_queuing_stats.
        """
        carried = defaultdict(dict)
        for ue in self.ues.values():
            if ue.is_active and ue.assigned_uav is not None:
                carried[ue.assigned_uav][ue.id] = min(self.throughput_cache[ue.id], getattr(ue, 'demand_bps', 0.0))

        for uav_id, uav in self.uavs.items():
            total = sum(carried[uav_id].values())
            capacity = self._backhaul_capacity(uav)
            self.backhaul_capacity[uav_id] = capacity
            self.backhaul_load[uav_id] = total / capacity
            if total > capacity:
                for ue_id in carried[uav_id]:
                    self.throughput_cache[ue_id] *= capacity / total

    def _update_per_cache(self):
        """Update PER cache for all UE-RB pairs"""
        self.per_cache.clear()

        # Collect all active UEs with RBs
        active_ues = [
            (ue_id, ue) for ue_id, ue in self.ues.items()
            if ue.is_active and ue.assigned_rb and ue.assigned_uav is not None
        ]
        
        if not active_ues:
            self.per_cache_valid = True
            return

        for ue_id, ue in active_ues:
            per = self._calculate_packet_error_rate(ue)
            self.per_cache[ue.id] = per

    def _invalidate_sinr_cache(self):
        """Invalidate cache when network state changes"""
        self.sinr_cache_valid = False
    
    def _invalidate_throughput_cache(self):
        """Invalidate throughput cache when allocations change"""
        self.throughput_cache_valid = False 

    def _get_cached_sinr(self, ue: UE, uav: UAV) -> float:
        """Get SINR from cache, updating if necessary"""
        if not self.sinr_cache_valid:
            self.cache_misses += 1
            self._update_sinr_cache()
        
        cache_key = (ue.id, uav.id)
        if cache_key in self.sinr_cache:
            self.cache_hits += 1
            return self.sinr_cache[cache_key]
        else:
            # Fallback to single calculation if not in cache
            # This shouldn't happen often if cache is properly maintained
            # pass
            # print(f"Cache miss for UE {ue.id}, UAV {uav.id}, RB {rb.id}")
            self.cache_misses += 1
            self._invalidate_sinr_cache()
            self._update_sinr_cache()
            return self.sinr_cache[cache_key]

    def _get_cached_delay(self, ue: UE) -> Dict[str, float]:
        # Return cached value
        if ue.id in self.delay_cache:
            return self.delay_cache[ue.id]
        else:
            self._update_delay_cache()
            return self.delay_cache[ue.id]
        
        # # Fallback for edge cases
        # return {'total': float('inf'), 'breakdown': {}}

    def _get_cached_per(self, ue: UE) -> float:
        # Return cached value
        if ue.id in self.per_cache:
            return self.per_cache[ue.id]
        else:
            self._update_per_cache()
            return self.per_cache[ue.id]
    
    def _get_cached_throughput(self, ue: UE) -> float:
        # Return cached value
        if ue.id in self.throughput_cache and self.throughput_cache_valid:
            return self.throughput_cache[ue.id]
        else:
            self._update_throughput_cache()
            return self.throughput_cache[ue.id]

    def _calculate_packet_error_rate(self, ue: UE) -> float:
        """Calculate PER based on SINR"""

        uav = self.uavs[ue.assigned_uav]
        avg_sinr_db = self._get_cached_sinr(ue, uav)
        packet_size_bytes = ue.packet_bytes


        # Approximate BLER curve
        if avg_sinr_db < -5:
            per = 0.5
        elif avg_sinr_db < 0:
            per = 0.2 * np.exp(-avg_sinr_db / 3)
        elif avg_sinr_db < 10:
            per = 0.1 * np.exp(-avg_sinr_db / 5)
        elif avg_sinr_db < 20:
            per = 0.01 * np.exp(-avg_sinr_db / 10)
        else:
            per = 1e-5
        
        # Adjust for packet size
        size_factor = packet_size_bytes / 1500.0
        return min(1.0, per * size_factor)

    def _allocate_rbs_fairly(self):
        """
        GLOBALLY OPTIMIZED: Allocate RBs for ALL UAVs/DAs/UEs at once
        - Processes all UAVs in parallel using vectorization
        - Single pass through all entities
        - Minimal object lookups
        """
        
        # ============================================
        # STEP 1: Reset all allocations (vectorized)
        # ============================================
        # Reset all RBs across all UAVs
        for uav in self.uavs.values():
            for rb in uav.RBs:
                rb.allocated_da_id = -1
                rb.allocated_ue_id = -1
        
        # Reset all DA RB lists
        for da in self.demand_areas.values():
            da.RB_ids_list = []
        
        # Reset all UE RB assignments
        for ue in self.ues.values():
            if ue.is_active:
                ue.assigned_rb = []
        
        # ============================================
        # STEP 2: Group DAs by UAV (pre-compute)
        # ============================================
        uav_das_map = defaultdict(list)  # {uav_id: [da1, da2, ...]}
        
        for da in self.demand_areas.values():
            uav_das_map[da.uav_id].append(da)
        
        # ============================================
        # STEP 3: Process each UAV (can be parallelized)
        # ============================================
        for uav_id, uav in self.uavs.items():
            uav_das = uav_das_map.get(uav_id, [])
            
            if not uav_das:
                continue
            
            total_rbs = len(uav.RBs)
            
            # Vectorized bandwidth to RB conversion
            allocated_bandwidths = np.array(
                [da.allocated_bandwidth for da in uav_das], 
                dtype=np.float32
            )
            
            target_rbs = (allocated_bandwidths / self.rb_bandwidth).astype(np.int32)
            
            # Ensure we don't exceed total RBs
            total_requested = target_rbs.sum()
            
            if total_requested > total_rbs:
                # Scale down proportionally
                target_rbs = ((target_rbs * total_rbs) / total_requested).astype(np.int32)
            
            # Allocate using cumulative sum
            cumsum = np.concatenate([[0], np.cumsum(target_rbs)])
            
            # ============================================
            # STEP 4: Assign RBs to DAs (vectorized)
            # ============================================
            for i, da in enumerate(uav_das):
                start_idx = cumsum[i]
                end_idx = min(cumsum[i + 1], total_rbs)
                
                # Batch assign RBs to this DA
                for rb_id in range(start_idx, end_idx):
                    rb = uav.RBs[rb_id]
                    rb.allocated_da_id = da.id
                    da.RB_ids_list.append(rb_id)
        
        # ============================================
        # STEP 5: Assign RBs to UEs (global pass)
        # ============================================
        # Build UE lookup for faster access
        active_ues = {ue_id: ue for ue_id, ue in self.ues.items() if ue.is_active}
        
        for da in self.demand_areas.values():
            # The scheduler serves only UEs with traffic this step (an idle eMBB UE gets nothing)
            users = [ue_id for ue_id in da.user_ids
                     if ue_id in active_ues and getattr(active_ues[ue_id], 'demand_bps', 0.0) > 0]
            if not users or not da.RB_ids_list:
                continue
            
            num_users = len(users)
            uav = self.uavs[da.uav_id]
            
            # Round-robin assignment
            for i, rb_id in enumerate(da.RB_ids_list):
                ue_id = users[i % num_users]
                rb = uav.RBs[rb_id]
                rb.allocated_ue_id = ue_id
                
                # Use pre-built lookup
                if ue_id in active_ues:
                    active_ues[ue_id].assigned_rb.append(rb)
        
        # ============================================
        # STEP 6: Invalidate cache once at the end
        # ============================================
        self._invalidate_sinr_cache()

    # ============================================
    # REWARD CALCULATION METHODS
    # ============================================

    def calculate_vector_reward(self) -> np.ndarray:
        """
        [MORL] Calculate reward vector for Multi-Objective RL.

        Returns:
            reward_vector: np.ndarray of shape (3,) with components:
                [0] QoS reward         ∈ [0, 1]   (maximize)
                [1] -Energy penalty    ∈ [-1, 0]  (maximize, i.e. minimize energy)
                [2] Fairness reward    ∈ [0, 1]   (maximize)
        """
        qos_reward, _ = self._calculate_qos_reward()
        energy_penalty = self._calculate_energy_consumption_penalty()
        fairness_reward = self._calculate_fairness_index()
        return np.array([qos_reward, -energy_penalty, fairness_reward], dtype=np.float32)

    def _get_global_observation(self) -> np.ndarray:
        """
        [MORL] Get concatenated observation of ALL UAVs as a single vector.
        Used by the single-agent MORL actor.

        Returns:
            global_obs: np.ndarray of shape (num_uavs * obs_dim_per_uav,)
        """
        per_uav_obs = self._get_observations()  # Dict[uav_id -> np.ndarray]
        # Concatenate in deterministic UAV order
        return np.concatenate(
            [per_uav_obs[uav_id] for uav_id in sorted(per_uav_obs.keys())],
            axis=0
        ).astype(np.float32)

    def calculate_global_reward(self) -> Tuple[float, float, float, float]:
        """
        REFACTORED: Calculate global reward with separated QoS calculation
        
        Returns:
            total_reward: Combined reward
            qos_reward: QoS satisfaction component
            energy_penalty: Energy consumption component
            fairness_reward: Fairness component
        """
        # ============================================
        # Calculate QoS reward (now separated!)
        # ============================================
        qos_reward, breakdown_qos = self._calculate_qos_reward()
        energy_penalty = self._calculate_energy_consumption_penalty()
        fairness_reward = self._calculate_fairness_index()
        
        # ============================================
        # Combine with weights
        # ============================================
        alpha = self.reward_weights.qos
        beta = self.reward_weights.energy
        gamma = self.reward_weights.fairness

        total_reward = (
            alpha * qos_reward -
            beta * energy_penalty +
            gamma * fairness_reward
        )

        breakdown = {
            'qos_reward': qos_reward,
            'energy_penalty': energy_penalty,
            'fairness_reward': fairness_reward,
            'avg_throughput_sat': breakdown_qos['avg_throughput_sat'],
            'avg_delay_sat': breakdown_qos['avg_delay_sat'],
            'avg_reliability_sat': breakdown_qos['avg_reliability_sat'],
        }

        return total_reward, breakdown

    def _calculate_qos_reward(self) -> Tuple[float, Dict]:
        """
        Calculate QoS reward for all UEs calculated based on throughput, delay, reliability.
        
        Returns:
            qos_reward: Aggregated QoS satisfaction score
            da_metrics: Detailed metrics per demand area
        """
        # Ensure SINR cache is valid
        if not self.sinr_cache_valid:
            self._update_sinr_cache()
        

        slice_weight_per_ue = []
        throughput_sats = []
        delay_sats = []
        reliability_sats = []
        overall_satisfactions = []

        # ============================================
        # Collect metrics for all active UEs
        # ============================================
        slice_of_ue = []
        for ue in self.ues.values():
            if not ue.is_active or getattr(ue, 'demand_bps', 0.0) <= 0:
                continue  # an idle eMBB UE has no demand this step
            
            # Slice weight of UE
            slice_weight_per_ue.append(self.slice_weights[ue.slice_type])
            slice_of_ue.append(ue.slice_type)

            # Calculate individual QoS components
            throughput_sat, ue_throughput = self._calculate_throughput_satisfaction(ue)
            delay_sat, total_delay_ms = self._calculate_delay_satisfaction(ue)
            reliability_sat, ue_reliability = self._calculate_reliability_satisfaction(ue)

            throughput_sats.append(throughput_sat)
            delay_sats.append(delay_sat)
            reliability_sats.append(reliability_sat)

            # Combine using slice-specific weights
            ue_qos_weights = self.qos_weights[ue.slice_type]
            overall_satisfaction = (
                ue_qos_weights['throughput'] * throughput_sat +
                ue_qos_weights['delay'] * delay_sat +
                ue_qos_weights['reliability'] * reliability_sat
            )

            overall_satisfactions.append(overall_satisfaction)

        # ============================================
        # Aggregate QoS reward across all DAs
        # ============================================

        # Mean within each slice, then the slice weights over the slices present: how many UEs a slice has
        # sets its load, not its share of the objective
        overall_satisfactions = np.array(overall_satisfactions, dtype=np.float64)
        slice_of_ue = np.array(slice_of_ue)
        present = [st for st in ('embb', 'urllc', 'mmtc') if np.any(slice_of_ue == st)]
        if present:
            qos_reward = sum(self.slice_weights[st] * overall_satisfactions[slice_of_ue == st].mean() for st in present)
            qos_reward /= sum(self.slice_weights[st] for st in present)
        else:
            qos_reward = 0.0

        breakdown = {
            'avg_throughput_sat': np.mean(throughput_sats) if len(throughput_sats) > 0 else 0.0,
            'avg_delay_sat': np.mean(delay_sats) if len(delay_sats) > 0 else np.inf,
            'avg_reliability_sat': np.mean(reliability_sats) if len(reliability_sats) > 0 else 0.0,
        }

        return qos_reward, breakdown

    def _calculate_throughput_satisfaction(self, ue) -> Tuple[float, float]:
        """
        Calculate throughput satisfaction for a single UE
        
        Returns:
            satisfaction: Normalized satisfaction score [0, 1]
            throughput: Actual throughput in bps
        """
        if not ue.assigned_rb or len(ue.assigned_rb) == 0:
            return 0.0, 0.0
        
        # Calculate total throughput across all RBs
        ue_throughput = self._get_cached_throughput(ue)
        
        # Normalize by minimum required rate. mMTC reports (tens of bytes every hour or so) need no rate:
        # its throughput term is whether the device is served at all
        min_rate = self.qos_profiles[ue.slice_type].min_rate
        satisfaction = 1.0 if min_rate <= 0 else min(ue_throughput / min_rate, 1.0)

        ue.throughput_satisfaction = satisfaction
        ue.throughput = ue_throughput
        
        return satisfaction, ue_throughput

    def _calculate_delay_satisfaction(self, ue) -> Tuple[float, float]:
        """
        Calculate delay satisfaction for a single UE
        
        Returns:
            satisfaction: Normalized satisfaction score [0, 1]
            delay_ms: Actual end-to-end delay in milliseconds
        """
        if not ue.assigned_rb or len(ue.assigned_rb) == 0:
            return 0.0, 99999.0
        
        # Get total delay
        delay_info = self._get_cached_delay(ue)
        total_delay_ms = delay_info['total']
        
        # Calculate satisfaction with exponential penalty
        max_latency = self.qos_profiles[ue.slice_type].max_latency
        
        if total_delay_ms <= max_latency:
            satisfaction = 1.0
        else:
            excess = total_delay_ms - max_latency
            satisfaction = np.exp(-excess / max_latency)

        ue.delay_satisfaction = satisfaction
        ue.latency_ms = total_delay_ms
        
        return satisfaction, total_delay_ms

    def _calculate_reliability_satisfaction(self, ue) -> float:
        """
        Calculate reliability satisfaction for a single UE
        
        Based on:
        - Buffer drops (from queuing model)
        - Handover losses
        - Packet error rate (from SINR)
        
        Returns:
            satisfaction: Normalized satisfaction score [0, 1]
        """
        if not ue.assigned_rb or len(ue.assigned_rb) == 0:
            return 0.0, 0.0
        
        # Drop rate (at assigned UAV)
        drop_rate = self.uavs[ue.assigned_uav].avg_drop_rate if ue.assigned_uav in self.uavs else 0.0
        
        # Handover loss
        # Only the packets in the interruption are at risk, not the whole T_L
        interrupted_share = getattr(ue, 'handover_penalty_ms', 0) / (self.T_L * 1000.0)
        handover_loss = self.handover_tracker.handover_failure_rate * interrupted_share
        
        # Transmission success based on PER
        per = self._get_cached_per(ue)
        transmission_success = (1 - per) ** 4  # With 4 retransmissions
        
        # Overall reliability
        reliability = transmission_success * (1 - drop_rate) * (1 - handover_loss)
        
        # Normalize
        min_reliability = self.qos_profiles[ue.slice_type].min_reliability
        satisfaction = min(reliability / min_reliability, 1.0)

        ue.reliability = reliability
        ue.reliability_satisfaction = satisfaction
        
        return satisfaction, reliability

    def _calculate_urllc_penalty(self, da_metrics: Dict) -> float:
        """
        Calculate penalty for URLLC violations
        
        Args:
            da_metrics: Metrics dictionary from _calculate_qos_reward
        
        Returns:
            penalty: Penalty score for URLLC violations
        """
        urllc_penalty = 0.0
        
        for da_id, metrics in da_metrics.items():
            if metrics['slice_type'] == 'urllc':
                # Penalize if any URLLC UE exceeds 1ms
                delays_over_threshold = [d for d in metrics['delays'] if d > 1.0]
                if delays_over_threshold:
                    violation_rate = len(delays_over_threshold) / len(metrics['delays'])
                    urllc_penalty += 0.5 * violation_rate
        
        return urllc_penalty

    def _get_qos_weights(self, slice_type: str) -> Dict[str, float]:
        """Get QoS metric weights per slice"""
        weights = {
            'embb': {
                'throughput': 0.7,
                'delay': 0.2,
                'reliability': 0.1
            },
            'urllc': {
                'throughput': 0.1,
                'delay': 0.5,      # CRITICAL
                'reliability': 0.4  # CRITICAL
            },
            'mmtc': {
                'throughput': 0.2,
                'delay': 0.1,
                'reliability': 0.7
            }
        }
        return weights[slice_type]

    def _propulsion_power(self, v_h: float, v_z: float, airspeed: Optional[float] = None) -> float:
        """
        Rotary-wing propulsion power in Watts plus climb power. Without airspeed: Zeng, Xu & Zhang 2019, Eq. 12.
        With it (a wind section): Dai et al. Eq. 8 at zero acceleration. The drag 0.5 rho S_FP |v - v_w|^2,
        with S_FP = d0 s A, adds to the thrust, T = |drag - m g|, and the parasite term uses the airspeed.
        Blade-profile and induced power use the ground speed as in Dai et al., or the airspeed if
        wind.rotor_speed is 'air'. With T = W they reduce to Zeng's P0 and Pi terms
        """
        m = self.energy_models
        v0 = m['mean_induced_velocity']
        drag_coef = 0.5 * m['fuselage_drag_ratio'] * m['air_density'] * m['rotor_solidity'] * m['rotor_disc_area']
        if airspeed is None:
            thrust_ratio, v_rotor, v_air = 1.0, v_h, v_h
        else:
            v_air = airspeed
            thrust_ratio = np.sqrt(1.0 + (drag_coef * v_air ** 2 / m['uav_weight']) ** 2)
            rotor_on_air = 'rotor_speed' in self.wind_params and self.wind_params['rotor_speed'] == 'air'
            v_rotor = v_air if rotor_on_air else v_h
        r = thrust_ratio
        blade = m['blade_profile_power'] * (r + 3 * v_rotor**2 / m['rotor_tip_speed']**2) * np.sqrt(r)
        induced = m['induced_power'] * r * np.sqrt(np.sqrt(r**2 + v_rotor**4 / (4 * v0**4)) - v_rotor**2 / (2 * v0**2))
        parasite = drag_coef * v_air**3
        climb = m['uav_weight'] * max(v_z, 0.0)  # Descent is not credited back
        return blade + induced + parasite + climb

    def _radio_power(self, tx_power: float) -> float:
        """
        Power drawn by the radio for a total transmit power in Watts. With radio_static_power, the EARTH
        linear model N_TRX * (P0 + dp * P_out) with the load shared by the TRXs: N_TRX * P0 + dp * P_tx.
        Otherwise only the PA, P_tx / pa_efficiency
        """
        m = self.energy_models
        if 'radio_static_power' in m:
            return m['radio_static_power'] + m['radio_power_slope'] * tx_power
        return tx_power / m['pa_efficiency']

    def _init_energy_bounds(self):
        """Lowest and highest total power draw, used to normalize the energy objective"""
        speeds = np.linspace(0.0, self.uav_params.velocity_max, 69)
        if self.wind_params is None:
            level_flight = [self._propulsion_power(v, 0.0) for v in speeds]
        else:
            # Every ground speed and airspeed the wind limit allows: |ground - air| <= max wind speed
            max_wind = self.wind_params['max_speed']
            level_flight = [self._propulsion_power(v, 0.0, a) for v in speeds for a in speeds if abs(v - a) <= max_wind]
        self.min_total_power = min(level_flight) + self._radio_power(self.min_power_ratio * self.uav_params.max_power)
        self.max_total_power = (max(level_flight) + self._propulsion_power(0.0, self.velocity_max_vertical)
                                - self._propulsion_power(0.0, 0.0) + self._radio_power(self.uav_params.max_power))

    def _calculate_energy_consumption_penalty(self) -> float:
        """Average UAV power draw over T_L, min-max normalized to [0, 1]"""
        total_energy_ratio = 0.0

        for uav in self.uavs.values():
            total_power = sum(uav.energy_used.values()) / self.T_L
            total_energy_ratio += (total_power - self.min_total_power) / (self.max_total_power - self.min_total_power)

        return float(np.clip(total_energy_ratio / self.num_uavs, 0.0, 1.0))

    def _calculate_fairness_index(self) -> float:
        """PRODUCTION VERSION: Fast and clean"""
        if not self.sinr_cache_valid:
            self._update_sinr_cache()
        
        # Pre-compute min rates
        min_rates = {
            'embb': self.qos_profiles['embb'].min_rate,
            'urllc': self.qos_profiles['urllc'].min_rate,
            'mmtc': self.qos_profiles['mmtc'].min_rate
        }
        
        # Collect normalized throughputs
        throughputs = []

        for ue in self.ues.values():
            if not ue.is_active or getattr(ue, 'demand_bps', 0.0) <= 0:
                continue  # an idle eMBB UE has no demand this step

            # Uncovered UEs count with zero throughput, so dropping UEs can't raise fairness
            ue_throughput = self._get_cached_throughput(ue)

            if self.fairness_metric == 'fraction_of_demand':
                # Jain, Chiu & Hawe 1984, Sec. 5 item 6: x_i = min(a_i / d_i, 1), with the UE's offered
                # load as its demand. Allocating more than the demand does not make the user any happier
                throughputs.append(min(ue_throughput / ue.demand_bps, 1.0))
            else:
                throughputs.append(ue_throughput / min_rates[ue.slice_type])
        
        if not throughputs:
            return 0.0
        
        # NumPy Jain's fairness
        x = np.array(throughputs, dtype=np.float32)
        n = len(x)
        sum_x = np.sum(x)
        sum_x_sq = np.sum(x * x)
        
        if sum_x_sq == 0:
            return 0.0
        
        return float((sum_x * sum_x) / (n * sum_x_sq))

    # ============================================
    # STATISTICS METHODS
    # ============================================

    def print_cache_stats(self):
        """Print cache performance statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses > 0:
            hit_rate = self.cache_hits / total_accesses * 100
            print(f"SINR Cache Stats: Hits={self.cache_hits}, Misses={self.cache_misses}, Hit Rate={hit_rate:.1f}%")
            print(f"Cache Size: {len(self.sinr_cache)} entries")

    def get_da_details(self):
        """Return detailed table lines of each Demand Area's performance"""
        lines = []

        if not self.demand_areas:
            lines.append("No Demand Areas to display")
            return lines

        lines.append("=" * 60)
        lines.append("DEMAND AREA PERFORMANCE DETAILS")
        lines.append("=" * 60)

        # Header
        header = f"{'ID':<4} {'UAV':<4} {'Slice':<6} {'Distance':<8} {'#UEs':<5} {'Allocated BW':<13} {'RBs':<4} {'Raw Alloc':<10}"
        lines.append(header)
        lines.append("-" * 60)

        # Body
        for da in self.demand_areas.values():
            allocated_bw_mhz = da.allocated_bandwidth  # Convert to MHz
            raw_allocated_action = da.raw_allocated_action
            row = f"{da.id:<4} {da.uav_id:<4} {da.slice_type.upper():<6} {da.distance_level:<8} {len(da.user_ids):<5} {allocated_bw_mhz:<13.2f} {len(da.RB_ids_list):<4} {raw_allocated_action:<10.3f}"
            lines.append(row)

        lines.append("-" * 60)
        total_ues = sum(len(da.user_ids) for da in self.demand_areas.values())
        total_allocated_bw = sum(da.allocated_bandwidth for da in self.demand_areas.values()) / 1e6  # MHz
        summary = f"SUMMARY: {len(self.demand_areas)} DAs | Total UEs: {total_ues} | Total Allocated BW: {total_allocated_bw:.2f} MHz"
        lines.append(summary)
        lines.append("=" * 60)

        return lines

    def _print_statistics(self):
        """Print current statistics of the environment"""

        print("Environment Initialization Parameters:")
        print(f"  Number of UAVs: {self.num_uavs}")
        print(f"  Number of UEs: {self.num_ues}")
        print(f"  Service Area: {self.service_area} meters")
        print(f"  UAV Flight Range X: {self.uav_fly_range_x} meters")
        print(f"  UAV Flight Range Y: {self.uav_fly_range_y} meters")
        print(f"  UAV Flight Range H: {self.uav_fly_range_h} meters")
        print(f"  UAV Beam Angle: {self.uav_beam_angle} degrees")
        print(f"  Number of DAs per UAV: {self.num_das_per_slice}")
        print(f"  Long-term period T_L: {self.T_L} seconds")
        print(f"  Short-term period T_S: {self.T_S} seconds")
        print(f"  QoS Profiles: {self.qos_profiles}")
        print(f"  Slice Weights: {self.slice_weights}")
        print(f"  Noise Power: {self.noise_power} Watts")
        print(f"  Carrier Frequency: {self.carrier_frequency} Hz")
        print(f"  RB Bandwidth: {self.rb_bandwidth} Hz")
        print(f"  Total Bandwidth per UAV: {self.total_bandwidth} Hz")
        print(f"  Total Resource Blocks: {self.total_rbs}")
        print(f"  UE Arrival Rate: {self.ue_arrival_rate} per second")
        print(f"  UE Departure Rate: {self.ue_departure_rate} per T_L")
        print(f"  UE dynamics Params: {self.ue_dynamics}")
        print(f"  Max UEs Allowed: {self.max_ues}")

    def get_uavs_power_usage(self) -> Dict[int, float]:
        """Get current power usage of each UAV"""
        uav_power_usage = {}
        for uav in self.uavs.values():
            uav_power_usage[uav.id] = uav.current_power
        return uav_power_usage

    def get_UEs_throughput_demand_and_satisfaction(self) -> Dict[int, Tuple[float, float, float]]:
        """Get current throughput, demand, and satisfaction of each UE"""
        ue_info = {}
        for ue in self.ues.values():
            if not ue.is_active:
                continue
            ue_throughput = self._get_cached_throughput(ue)

            min_rate = self.qos_profiles[ue.slice_type].min_rate
            satisfaction = min(ue_throughput / min_rate, 1.0) if min_rate > 0 else 1.0
            
            ue_info[ue.id] = (ue_throughput, min_rate, satisfaction)
        
        return ue_info