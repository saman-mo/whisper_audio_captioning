aws_region         = "eu-central-1"
cluster_name       = "whisper-mesh-dev"
use_spot_instances = true
cpu_instance_types = ["t3.large", "t3a.large"]
cpu_desired_size   = 2
enable_gpu_nodes   = false
