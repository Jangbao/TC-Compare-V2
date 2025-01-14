# TC-Compare-V2
Performance Evaluation of Triangle Counting Algorithms on GPUs

## Quick Start

### **Build the Project**

To compile the project, navigate to the `build` directory and run the following commands:

```bash
cd build 
cmake ..
make -j
```

**Run a Test**
Execute the following command to test the default graph Com-Dblp:

```
./main
```

If you see output similar to the following, the setup is complete. Congratulations!

```
[2025-01-14 16:31:42.852] [info] Run algorithm GroupTC-BS
[2025-01-14 16:31:42.852] [info] Dataset /home/LiJB/cuda_project/TC-compare-V100/dataset_app/csr_dataset/Com-Dblp
[2025-01-14 16:31:42.852] [info] Number of nodes: 317080, number of edges: 1049866
[2025-01-14 16:31:42.954] [info] Iter 0, kernel use 0.000164 s
[2025-01-14 16:31:43.803] [info] iter 10, avg kernel use 0.000154 s
[2025-01-14 16:31:43.803] [info] Triangle count 2224385
[2025-01-14 16:31:44.032] [info] Run algorithm GroupTC-HS
[2025-01-14 16:31:44.032] [info] Dataset /home/LiJB/cuda_project/TC-compare-V100/dataset_app/csr_dataset/Com-Dblp
[2025-01-14 16:31:44.032] [info] Number of nodes: 317080, number of edges: 1049866
[2025-01-14 16:31:44.035] [info] kernel_grid_size 2048
[2025-01-14 16:31:44.113] [info] Iter 0, kernel use 0.000192 s
[2025-01-14 16:31:44.801] [info] Iter 10, avg kernel use 0.000188 s
[2025-01-14 16:31:44.801] [info] Triangle count 2224385
```


**Download more graphs from SNAP**
To test with more graphs, download graphs from the SNAP Repository(https://snap.stanford.edu/data/):

```
cd data_getter/snap
./WgetSNAPData.sh
```

You can add new graphs by modifying the `WgetSNAPData.sh` script. For example:
```
mkdir data/snap_dataset/Com-Orkut/
wget -P ../../data/snap_dataset/Com-Orkut/   https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
gzip -d ../../data/snap_dataset/Com-Orkut/com-orkut.ungraph.txt.gz
```

Downloaded files are in text format and must be converted to CSR (Compressed Sparse Row) format using preprocessing tools:
```
mkdir data/csr_dataset/Com-Orkut/
cd preprocessing/XXX2CSR
g++ SNAP2CSR.cpp -o SNAP2CSR
./SNAP2CSR /{your project path}/data/snap_dataset/Com-Orkut/com-orkut.ungraph.txt \
/{your project path}/data/csr_dataset/Com-Orkut/
```

## Performance

The following table summarizes the performance of triangle counting algorithms on different graphs:

| Datasets            | ABBR. | GroupTC-BS (ms) | GroupTC-HS (ms) |
|---------------------|-------|-----------------|-----------------|
| Web-NotreDame       | WN    | 0.24            | 0.24            |
| Com-Dblp            | CD    | 0.15            | 0.19            |
| Amazon0601          | AM    | 0.27            | 0.38            |
| RoadNet-CA          | RC    | 0.23            | 0.36            |
| Wiki-Talk           | WT    | 0.92            | 1.16            |
| Imdb-2021           | IM    | 1.31            | 1.09            |
| Web-BerkStan        | WB    | 1.53            | 1.22            |
| As-Skitter          | AS    | 2.27            | 2.21            |
| Cit-Patents         | CP    | 4.17            | 4.61            |
| Soc-Pokec           | SP    | 8.16            | 7.15            |
| Sx-Stackoverflow    | SX    | 20.30           | 21.55           |
| Com-Lj              | CL    | 13.36           | 9.74            |
| Soc-LiveJournal     | SL    | 18.32           | 13.98           |
| k-mer-graph5        | K5    | 4.93            | 4.56            |
| Hollywood-2011      | HW    | 251.08          | 184.27          |
| Com-Orkut           | CO    | 93.94           | 77.62           |
| Enwiki-2024         | EN    | 97.74           | 78.88           |
| k-mer-graph4        | K4    | 15.57           | 17.42           |
| Twitter7            | TW    | 3,048.39        | 2,887.75        |
| Com-Friendster      | CF    | 2,166.50        | 1,932.17        |

