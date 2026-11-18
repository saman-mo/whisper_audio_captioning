
variable "aws_region" {
  description = "AWS region to deploy the cluster (Frankfurt)"
  type        = string
  default     = "eu-central-1"
}

variable "cluster_name" {
  description = "EKS cluster name (used to isolate dev vs prod resources)"
  type        = string
  default     = "whisper-mesh-dev"
}

variable "kubernetes_version" {
  description = "EKS Kubernetes version"
  type        = string
  default     = "1.31"
}

variable "use_spot_instances" {
  description = "Use EC2 Spot instances for cheap demo/showcase hosting"
  type        = bool
  default     = true
}

variable "cpu_instance_types" {
  description = "Instance types for the primary CPU worker pool"
  type        = list(string)
  default     = ["t3.large", "t3a.large"]
}

variable "cpu_desired_size" {
  description = "Desired number of CPU worker nodes"
  type        = number
  default     = 1
}

variable "enable_gpu_nodes" {
  description = "Toggle GPU worker pool (false = $0 GPU cost; true = provisions g5.xlarge)"
  type        = bool
  default     = false
}

variable "gpu_instance_types" {
  description = "GPU instance types when enable_gpu_nodes is true"
  type        = list(string)
  default     = ["g5.xlarge"]
}
