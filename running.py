from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

job_array = [
    "python -O train_representation.py runs/MIB --config-file=configs/MIB.yml --data-dir=./data/",
    "python -O train_representation.py runs/InfoMax --config-file=configs/InfoMax.yml --data-dir=./data/",
    "python -O train_representation.py runs/MV_InfoMax --config-file=configs/MV_InfoMax.yml --data-dir=./data/",
    "python -O train_representation.py runs/VAE --config-file=configs/VAE.yml --data-dir=./data/",

    "python -O train_representation.py runs/IIC --config-file=configs/IIC.yml --data-dir=./data/",
    "python -O train_representation.py runs/IIC_mib --config-file=configs/IIC_mib.yml --data-dir=./data/",
    "python -O train_representation.py runs/IIC_resample --config-file=configs/IIC_resample.yml --data-dir=./data/",

    "python -O train_representation.py runs/InfoNCE --config-file=configs/InfoNCE.yml --data-dir=./data/",
    "python -O train_representation.py runs/InfoNCE_resample --config-file=configs/InfoNCE_resample.yml --data-dir=./data/"
]

accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="./", on_local=False, time=3)
for j in job_array:
    jobsubmiter.prepare_env(
        [
            "source ./venv/bin/activate ",
            "export OMP_NUM_THREADS=1",
            "export PYTHONOPTIMIZE=1"
        ]
    )
    jobsubmiter.account = next(accounts)
    jobsubmiter.run(j)
