(venv) ubuntu@129-213-25-237:~/xnotes/communitynotes-mod/scoring/src$   python main.py \
    --enrollment data/userEnrollment-00000.tsv \
    --notes data/notes-00000.tsv \
    --ratings data/ratings \
    --status data/noteStatusHistory-00000.tsv \
    --outdir data \
    --cache-dir .cache
00:12:16 INFO:birdwatch.runner:scorer python version: 3.10.12 (main, Aug 15 2025, 14:32:43) [GCC 11.4.0]
00:12:16 INFO:birdwatch.runner:scorer pandas version: 2.2.2
00:12:16 INFO:birdwatch.dev_cache:Dev cache enabled: /lambda/nfs/xnotes/communitynotes-mod/scoring/src/.cache
Patching pandas
00:12:16 INFO:birdwatch.runner:beginning scorer execution
00:13:27 INFO:birdwatch.dev_cache:Cache hit: 'runner_data' (loaded in 71.9s) <- /lambda/nfs/xnotes/communitynotes-mod/scoring/src/.cache/runner_data.pkl
00:13:28 INFO:birdwatch.process_data:Called filter_input_data_for_testing.
        Notes: 2380443, Ratings: 197106676. Max note createdAt: 2026-02-01 23:01:46.167000; Max rating createAt: 2026-02-01 23:02:41.892000
00:13:28 INFO:birdwatch.process_data:After filtering notes and ratings after particular timestamp (=None). 
        Notes: 2380443, Ratings: 197106676. Max note createdAt: 2026-02-01 23:01:46.167000; Max rating createAt: 2026-02-01 23:02:41.892000
00:13:28 INFO:birdwatch.process_data:After filtering ratings after first status (plus None hours) for notes created in last 14 days. 
        Notes: 2380443, Ratings: 197106676. Max note createdAt: 2026-02-01 23:01:46.167000; Max rating createAt: 2026-02-01 23:02:41.892000
00:13:28 INFO:birdwatch.process_data:After filtering prescoring notes and ratings to simulate a delay of None hours: 
        Notes: 2380443, Ratings: 197106676. Max note createdAt: 2026-02-01 23:01:46.167000; Max rating createAt: 2026-02-01 23:02:41.892000
00:14:27 INFO:birdwatch.dev_cache:Cache hit: 'pss_pre_pair_counts' (loaded in 58.8s) <- /lambda/nfs/xnotes/communitynotes-mod/scoring/src/.cache/pss_pre_pair_counts.pkl
00:14:27 INFO:birdwatch.post_selection_similarity:Restored PSS intermediate state from cache — skipping to _get_pair_counts_dict
00:15:49 INFO:birdwatch.post_selection_similarity:Computed unique ratings on tweets for 123472011 ratings
00:16:07 INFO:birdwatch.post_selection_similarity:Pre-filter threshold: minCoRatingCount=8 (minSim=8, npmi=8)
00:16:07 INFO:birdwatch.post_selection_similarity:Starting pair counts computation (Numba + array)
00:16:07 INFO:birdwatch.post_selection_similarity:[mem] start: RSS 31.3GB, available: 186.5GB
00:16:55 INFO:birdwatch.post_selection_similarity:[mem] after extracting arrays: RSS 34.6GB, available: 183.0GB
00:17:34 INFO:birdwatch.post_selection_similarity:[mem] after preprocessing: 179,533,354 ratings, 2,152,153 groups: RSS 33.3GB, available: 184.6GB
00:17:34 INFO:birdwatch.post_selection_similarity:Counting pair events...
00:17:38 INFO:birdwatch.post_selection_similarity:Total pair events: 2,199,474,894
00:17:38 INFO:birdwatch.post_selection_similarity:[mem] after count pass: RSS 33.3GB, available: 184.7GB
00:17:38 INFO:birdwatch.post_selection_similarity:Generating pair events...
00:17:38 INFO:birdwatch.post_selection_similarity:[mem] after allocating output arrays: RSS 33.3GB, available: 184.7GB
00:17:49 INFO:birdwatch.post_selection_similarity:[mem] after fill pass: RSS 55.9GB, available: 161.7GB
00:17:49 INFO:birdwatch.post_selection_similarity:Sorting pair events...
00:31:22 INFO:birdwatch.post_selection_similarity:[mem] after sort: RSS 55.9GB, available: 162.7GB
00:31:42 INFO:birdwatch.post_selection_similarity:After per-tweet dedup: 1,800,891,157 unique pair-tweet events
00:31:42 INFO:birdwatch.post_selection_similarity:[mem] after dedup: RSS 44.8GB, available: 173.6GB
00:32:01 INFO:birdwatch.post_selection_similarity:Total unique pairs: 1,528,653,805
00:32:01 INFO:birdwatch.post_selection_similarity:Pairs with count >= 8: 1,528,653,805
00:32:01 INFO:birdwatch.post_selection_similarity:[mem] after filtering: RSS 65.5GB, available: 152.4GB
Killed


(venv) ubuntu@129-213-25-237:~/xnotes/communitynotes-mod/scoring/src$   python main.py \
    --enrollment data/userEnrollment-00000.tsv \
    --notes data/notes-00000.tsv \
    --ratings data/ratings \
    --status data/noteStatusHistory-00000.tsv \
    --outdir data \
    --cache-dir .cache
01:28:08 INFO:birdwatch.runner:scorer python version: 3.10.12 (main, Aug 15 2025, 14:32:43) [GCC 11.4.0]
01:28:08 INFO:birdwatch.runner:scorer pandas version: 2.2.2
01:28:08 INFO:birdwatch.dev_cache:Dev cache enabled: /lambda/nfs/xnotes/communitynotes-mod/scoring/src/.cache
Patching pandas
01:28:08 INFO:birdwatch.runner:beginning scorer execution
01:28:28 INFO:birdwatch.dev_cache:Cache hit: 'runner_data' (loaded in 20.1s) <- /lambda/nfs/xnotes/communitynotes-mod/scoring/src/.cache/runner_data.pkl
01:28:28 INFO:birdwatch.process_data:Called filter_input_data_for_testing.
        Notes: 2380443, Ratings: 197106676. Max note createdAt: 2026-02-01 23:01:46.167000; Max rating createAt: 2026-02-01 23:02:41.892000
01:28:28 INFO:birdwatch.process_data:After filtering notes and ratings after particular timestamp (=None). 
        Notes: 2380443, Ratings: 197106676. Max note createdAt: 2026-02-01 23:01:46.167000; Max rating createAt: 2026-02-01 23:02:41.892000
01:28:28 INFO:birdwatch.process_data:After filtering ratings after first status (plus None hours) for notes created in last 14 days. 
        Notes: 2380443, Ratings: 197106676. Max note createdAt: 2026-02-01 23:01:46.167000; Max rating createAt: 2026-02-01 23:02:41.892000
01:28:28 INFO:birdwatch.process_data:After filtering prescoring notes and ratings to simulate a delay of None hours: 
        Notes: 2380443, Ratings: 197106676. Max note createdAt: 2026-02-01 23:01:46.167000; Max rating createAt: 2026-02-01 23:02:41.892000
01:28:48 INFO:birdwatch.dev_cache:Cache hit: 'pss_pre_pair_counts' (loaded in 19.8s) <- /lambda/nfs/xnotes/communitynotes-mod/scoring/src/.cache/pss_pre_pair_counts.pkl
01:28:48 INFO:birdwatch.post_selection_similarity:Restored PSS intermediate state from cache — skipping to _get_pair_counts_dict
01:30:10 INFO:birdwatch.post_selection_similarity:Computed unique ratings on tweets for 123472011 ratings
01:30:28 INFO:birdwatch.post_selection_similarity:Pre-filter threshold: minCoRatingCount=8 (minSim=8, npmi=8)
01:30:28 INFO:birdwatch.post_selection_similarity:Starting pair counts computation (Numba + array)
01:30:28 INFO:birdwatch.post_selection_similarity:[mem] start: RSS 31.3GB, available: 187.1GB
01:31:15 INFO:birdwatch.post_selection_similarity:[mem] after extracting arrays: RSS 34.6GB, available: 183.3GB
01:31:54 INFO:birdwatch.post_selection_similarity:[mem] after preprocessing: 179,533,354 ratings, 2,152,153 groups: RSS 33.3GB, available: 184.4GB
01:31:54 INFO:birdwatch.post_selection_similarity:Counting pair events...
01:31:57 INFO:birdwatch.post_selection_similarity:Total pair events: 2,199,474,894
01:31:57 INFO:birdwatch.post_selection_similarity:[mem] after count pass: RSS 33.3GB, available: 184.5GB
01:31:57 INFO:birdwatch.post_selection_similarity:Generating pair events...
01:31:57 INFO:birdwatch.post_selection_similarity:[mem] after allocating output arrays: RSS 33.3GB, available: 184.5GB
01:32:09 INFO:birdwatch.post_selection_similarity:[mem] after fill pass: RSS 55.9GB, available: 161.2GB
01:32:09 INFO:birdwatch.post_selection_similarity:Sorting pair events...
01:45:40 INFO:birdwatch.post_selection_similarity:[mem] after sort: RSS 55.9GB, available: 162.7GB
01:46:00 INFO:birdwatch.post_selection_similarity:After per-tweet dedup: 1,800,891,157 unique pair-tweet events
01:46:00 INFO:birdwatch.post_selection_similarity:[mem] after dedup: RSS 44.8GB, available: 173.7GB
01:46:18 INFO:birdwatch.post_selection_similarity:Total unique pairs: 1,528,653,805
01:46:21 INFO:birdwatch.post_selection_similarity:Pairs with count >= 8: 3,659,389
01:46:21 INFO:birdwatch.post_selection_similarity:[mem] after filtering: RSS 55.6GB, available: 162.6GB
01:46:26 INFO:birdwatch.post_selection_similarity:Output: 3,659,389 pairs
01:46:26 INFO:birdwatch.post_selection_similarity:[mem] done: RSS 56.0GB, available: 162.3GB
01:46:27 INFO:birdwatch.constants:Compute pair counts dict elapsed time: 958.72 secs (15.98 mins)
01:46:27 INFO:birdwatch.post_selection_similarity:Computed pair counts dict for 3659389 pairs
Pairs dict used 0.167772256GB RAM at max
01:46:36 INFO:birdwatch.constants:Compute PMI and minSim elapsed time: 9.13 secs (0.15 mins)
01:46:36 INFO:birdwatch.constants:Delete unneeded pairs from pairCountsDict elapsed time: 0.58 secs (0.01 mins)
Pairs dict used 0.167772256GB RAM after deleted unneeded pairs
01:46:36 INFO:birdwatch.post_selection_similarity:Computed suspect pairs for 6152 pairs
01:46:51 INFO:birdwatch.constants:Aggregate into cliques by post selection similarity elapsed time: 14.80 secs (0.25 mins)
01:46:51 INFO:birdwatch.post_selection_similarity:Aggregated into 536 cliques
01:46:51 INFO:birdwatch.post_selection_similarity:Computed cliques dataframe for 2591 cliques
01:46:55 INFO:birdwatch.constants:Compute Post Selection Similarity elapsed time: 1106.25 secs (18.44 mins)