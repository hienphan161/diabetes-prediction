// This will create the output likes this
// Apply complete! Resources: 2 added, 0 changed, 0 destroyed.

// Outputs:
//    kubernetes_cluster_host = "35.239.66.181"
//    kubernetes_cluster_name = "diabetes-prediction-gke"
//    project_id = "diabetes-prediction-483908"
//    region = "us-central1-c"

output "project_id" {
  value       = var.project_id
  description = "Project ID"
}

output "kubernetes_cluster_name" {
  value       = google_container_cluster.primary.name
  description = "GKE Cluster Name"
}

output "kubernetes_cluster_host" {
  value       = google_container_cluster.primary.endpoint
  description = "GKE Cluster Host"
}

output "region" {
  value       = var.region
  description = "GKE region"
}