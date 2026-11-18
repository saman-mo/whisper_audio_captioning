terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = "whisper-mesh"
      Cluster   = var.cluster_name
      ManagedBy = "Terraform"
    }
  }
}

data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
}

# =========================================================================
# 1. ZERO-TRUST VPC (Public ALB Subnets + Private EKS Worker Subnets)
# =========================================================================
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.13"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = local.azs
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true # Cost-optimized: 1 NAT Gateway instead of 1 per AZ
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Required tags for AWS Load Balancer Controller auto-discovery
  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}

# =========================================================================
# 2. EKS CLUSTER & MANAGED NODE GROUPS (CPU Default + Toggleable GPU)
# =========================================================================
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.24"

  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version

  cluster_endpoint_public_access           = true
  enable_cluster_creator_admin_permissions = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Core EKS Add-ons (including EBS CSI for persistent Ollama/Redis storage)
  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa.iam_role_arn
    }
  }

  eks_managed_node_groups = merge(
    # ---------------------------------------------------------------------
    # ALWAYS-ON COST-OPTIMIZED CPU NODE POOL
    # ---------------------------------------------------------------------
    {
      cpu_workers = {
        name           = "cpu-workers"
        ami_type       = "AL2023_x86_64_STANDARD"
        instance_types = var.cpu_instance_types
        capacity_type  = var.use_spot_instances ? "SPOT" : "ON_DEMAND"

        min_size     = 1
        max_size     = 4
        desired_size = var.cpu_desired_size

        disk_size = 50

        labels = {
          workload-type = "cpu"
        }
      }
    },
    # ---------------------------------------------------------------------
    # OPTIONAL GPU INFERENCE POOL (Provisions 0 resources unless toggled true)
    # ---------------------------------------------------------------------
    var.enable_gpu_nodes ? {
      gpu_inference = {
        name           = "gpu-inference"
        ami_type       = "AL2023_x86_64_NVIDIA"
        instance_types = var.gpu_instance_types
        capacity_type  = var.use_spot_instances ? "SPOT" : "ON_DEMAND"

        min_size     = 0
        max_size     = 2
        desired_size = 1

        disk_size = 100

        labels = {
          workload-type = "gpu"
        }

        taints = [
          {
            key    = "://nvidia.com"
            value  = "true"
            effect = "NO_SCHEDULE"
          }
        ]
      }
    } : {}
  )
}

# =========================================================================
# 3. PRIVATE ECR REPOSITORY FOR WHISPER-BACKEND IMAGE (SCOPED PER CLUSTER)
# =========================================================================
resource "aws_ecr_repository" "whisper_backend" {
  name                 = "${var.cluster_name}-backend"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # Allows clean terraform destroy after showcase

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "cleanup_old_images" {
  repository = aws_ecr_repository.whisper_backend.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 3 most recent images to minimize storage cost"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# =========================================================================
# 4. IRSA (IAM Roles for Service Accounts - Zero Hardcoded AWS Keys)
# =========================================================================
module "ebs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.44"

  role_name_prefix      = "${var.cluster_name}-ebs-csi-"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }
}

module "lb_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.44"

  role_name_prefix                       = "${var.cluster_name}-alb-ctrl-"
  attach_load_balancer_controller_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }
}
