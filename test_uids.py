import random
import objaverse
from objaverse_paths import configure_objaverse
configure_objaverse()

def main():
    print("Fetching Objaverse UIDs...")
    
    # 1. Mimic Precompute Phase (Seed 42)
    # The script loads all objaverse UIDs and shuffles them with seed 42.
    uids = list(objaverse.load_uids())
    rng_precompute = random.Random(42)
    rng_precompute.shuffle(uids)
    
    # We assume the first 100 meshes are used. 
    # NOTE: Your actual precompute script skips meshes that are too large or not watertight.
    # If any of these first 100 fail those checks during real execution, the list will shift.
    precomputed_uids = uids[:100]

    # 2. Mimic Train Phase Loading
    # run_train sorts the directory names alphabetically before doing anything else.
    all_uids = sorted(precomputed_uids)

    # 3. Train/Test UID Split (Seed 42 initialized again)
    rng_split = random.Random(42)
    shuffled_uids = list(all_uids)
    rng_split.shuffle(shuffled_uids)
    
    test_fraction = 0.2
    n_test = int(len(shuffled_uids) * test_fraction)
    
    test_uids = set(shuffled_uids[:n_test])
    train_uids = set(shuffled_uids[n_test:])

    # 4. Train/Test View Split (Seed 43)
    # Reconstructing the expected file names
    azimuths = [0.0, 72.0, 144.0, 216.0, 288.0]
    elevations = [15.0, 30.0]
    all_sample_names = []
    
    for uid in all_uids:
        for az in azimuths:
            for el in elevations:
                all_sample_names.append(f"{uid}_az{int(az):03d}_el{int(el):03d}")
                
    all_sample_names = sorted(all_sample_names)
    train_sample_names = [s for s in all_sample_names if s.split("_az")[0] in train_uids]
    
    test_view_names = set()
    test_view_fraction = 0.2
    
    if test_view_fraction > 0 and len(train_sample_names) > 1:
        rng_view = random.Random(43)
        rng_view.shuffle(train_sample_names)
        n_test_views = max(1, int(len(train_sample_names) * test_view_fraction))
        test_view_names = set(train_sample_names[:n_test_views])
        train_view_names = set(train_sample_names[n_test_views:])
    else:
        train_view_names = set(train_sample_names)

    # 5. Wandb Vis Split (Seed 0)
    def _pick_vis_from_uids(uid_set_in, n):
        if not uid_set_in: return []
        return random.Random(0).sample(sorted(uid_set_in), min(n, len(uid_set_in)))

    vis_seen_uids = _pick_vis_from_uids(train_uids, 3)
    vis_unseen_uids = _pick_vis_from_uids(test_uids, 3)

    # --- Output ---
    print(f"\n--- UIDs (Total: 100) ---")
    print(f"TRAIN UIDs ({len(train_uids)}):\n{sorted(train_uids)}\n")
    print(f"TEST UIDs ({len(test_uids)}):\n{sorted(test_uids)}\n")
    
    print(f"--- VIEWS (Total Candidate Train Views: {len(train_sample_names)}) ---")
    print(f"TRAIN VIEWS ({len(train_view_names)} total)")
    print(f"HELD-OUT TEST VIEWS ({len(test_view_names)} total)\n")
    
    print(f"--- WANDB VISUALIZATION ---")
    print(f"VIS SEEN UIDs: {vis_seen_uids}")
    print(f"VIS UNSEEN UIDs: {vis_unseen_uids}")

if __name__ == "__main__":
    main()