export type Mode = 'full_multimodal' | 'missing_feature_demo'

export interface ModelInfo {
  architecture: string
  pretrained_backbone: string
  checkpoint_path: string
  checkpoint_sha256: string
  parameter_count: number
  device: string
  sequence_length_tokens: number
}

export interface FeatureStatus { available: boolean; provenance: string }

export interface PredictionResponse {
  input_mode: Mode
  predicted_cleavage_frequency: number
  high_efficiency_probability: number
  high_efficiency_prediction: boolean
  sequence_nucleotides: number
  sequence_tokens_used: number
  sequence_length_tokens: number
  model: ModelInfo
  feature_availability: Record<string, FeatureStatus>
  warning: string | null
}

export interface HealthResponse { status: 'ready' | 'not_ready'; device: string | null; error: string | null }

export interface ModelInfoResponse extends ModelInfo { status: 'loaded'; epigenetic_scaler_available: boolean }
