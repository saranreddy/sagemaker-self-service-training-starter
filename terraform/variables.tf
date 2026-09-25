variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name (used for naming resources)"
  type        = string
  default     = "sagemaker-self-service-training"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,40}[a-z0-9]$", var.project_name))
    error_message = "Project name must be 3-42 characters, start and end with alphanumeric, and contain only lowercase letters, numbers, and hyphens."
  }
}

variable "artifact_bucket_name" {
  description = "Name for the artifact S3 bucket (leave empty for auto-generated name)"
  type        = string
  default     = ""
}
