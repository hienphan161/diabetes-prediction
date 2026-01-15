// Variables to use accross the project
// which can be accessed by var.project_id
variable "project_id" {
  description = "The project ID to host the cluster in"
  default     = "diabetes-prediction-483908"
}

variable "project_name" {
  description = "The project name"
  default     = "diabetes-prediction"
}

variable "region" {
  description = "The region the cluster in"
  default     = "us-central1-c"
}