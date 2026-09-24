output "execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "artifact_bucket_name" {
  description = "Name of the artifact S3 bucket"
  value       = aws_s3_bucket.artifact_bucket.id
}

output "artifact_bucket_arn" {
  description = "ARN of the artifact S3 bucket"
  value       = aws_s3_bucket.artifact_bucket.arn
}

output "region" {
  description = "AWS region"
  value       = local.region
}

output "account_id" {
  description = "AWS account ID"
  value       = local.account_id
}

output "org_config_yaml" {
  description = "Suggested org-config.yaml content (standalone mode)"
  value       = <<-EOT
    # org-config.yaml - Generated from Terraform outputs
    # Place this file at the project root, ~/.mlctl/, or /etc/mlctl/
    # Image URIs are resolved automatically per region - no templates needed
    
    execution_role: ${aws_iam_role.sagemaker_execution_role.arn}
    artifact_bucket: ${aws_s3_bucket.artifact_bucket.id}
    
    frameworks:
      sklearn:
        default_instance_type: "ml.m5.large"
        version: "1.2-1"
      xgboost:
        default_instance_type: "ml.m5.large"
        version: "1.7-1"
      pytorch:
        default_instance_type: "ml.m5.large"
        version: "2.1.0"
    
    default_instance_type: "ml.m5.large"
    
    allowed_instance_types:
      - ml.m5.large
      - ml.m5.xlarge
      - ml.m5.2xlarge
      - ml.m5.4xlarge
      - ml.c5.xlarge
      - ml.c5.2xlarge
      - ml.c5.4xlarge
    
    required_tags:
      Project: ${var.project_name}
    
    teams: {}
  EOT
}
