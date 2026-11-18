output "cluster_name" {
  description = "EKS Cluster Name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS Control Plane Endpoint"
  value       = module.eks.cluster_endpoint
}

output "ecr_repository_url" {
  description = "Docker image push URL for whisper-backend"
  value       = aws_ecr_repository.whisper_backend.repository_url
}

output "lb_controller_role_arn" {
  description = "IAM Role ARN for AWS Load Balancer Controller Helm chart"
  value       = module.lb_controller_irsa.iam_role_arn
}

output "configure_kubectl" {
  description = "Command to connect local kubectl to EKS"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}
