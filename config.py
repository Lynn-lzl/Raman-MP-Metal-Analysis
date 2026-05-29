class Config:
    RANDOM_STATE = 42
    
    # Nested CV
    OUTER_SPLITS = 5
    INNER_SPLITS = 5
    
    # Upper limit for shared feature selection.
    # The per-target settings below override this value when present.
    SHARED_MAX_FEATURES = 13
    TARGET_SHARED_MAX_FEATURES = {
        "118Sn (KED)": 10,
        "209Bi (KED)": 13,
    }
    
    # physical partition (400-1800 nm)
    PHYSICAL_ZONE_EDGES = [400, 900, 1400, 1800]
    PHYSICAL_ZONE_NAMES = ["LFR(400-900)", "MFR(900-1400)", "HFR(1400-1800)"]
    
    # MoE zone filtering threshold
    MOE_ZONE_R2_THRESHOLD = 0.8
    
    # Position encoding hyperparameter
    USE_GLOBAL_COSPOS = True
    USE_ZONE_LOCAL_COSPOS = False
    GLOBAL_COS_OMEGAS = [1.0, 2.0, 4.0, 8.0]
    ZONE_COS_OMEGAS = [1.0, 2.0]
    ZONE_LOCAL_COS_MIN_SIZE = 3
    ZONE_LOCAL_COS_SELECTED = None
