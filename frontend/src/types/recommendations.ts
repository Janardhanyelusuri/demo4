// src/types/recommendations.ts

// Define the structure for a single recommendation (effective or additional)
export interface RecommendationDetail { // <-- FIX: Added 'export' here
  text: string;
  explanation?: string; // LLM-provided explanation for why this recommendation was made
  saving_pct: number;
}

// Define the main recommendations block
interface RecommendationsBlock {
  effective_recommendation: RecommendationDetail;
  additional_recommendation: RecommendationDetail[];
  base_of_recommendations: string[];
}

// Define the cost forecasting block
interface CostForecasting {
  monthly: number;
  annually: number;
}

// Define the anomaly block
interface Anomaly {
  metric_name: string;
  timestamp: string;
  value: number;
  reason_short: string;
}

// Define the contract deal block
interface ContractDeal {
  assessment: "good" | "bad" | "unknown";
  'for sku': string;
  reason: string;
  monthly_saving_pct: number;
  annual_saving_pct: number;
}

// Define the raw LLM response structure for a single resource
export interface RawRecommendation {
  recommendations: RecommendationsBlock;
  cost_forecasting: CostForecasting;
  anomalies: Anomaly[];
  contract_deal: ContractDeal;
  resource_id: string; // Added by the Python code
  _forecast_monthly?: number; // Added by the Python code
  _forecast_annual?: number; // Added by the Python code
}

// Define the normalized, ready-to-use structure for the frontend UI
export interface NormalizedRecommendation {
  // Resource info
  resourceId: string;

  // Recommendations
  effectiveRecommendation: RecommendationDetail;
  additionalRecommendations: RecommendationDetail[];
  baseOfRecommendations: string[];
  totalSavingPercent: number;

  // Forecasting
  costForecasting: CostForecasting;

  // Anomalies
  anomalies: Anomaly[];

  // Contract/Deal info
  contractDeal: ContractDeal;

  // Severity (for UI styling)
  severity: 'High' | 'Medium' | 'Low';
}

// --- NEW FILTER AND MAPPING TYPES ---

// Resource mapping structure for the UI
export interface CloudResourceMap {
    displayName: string; // e.g., "VM"
    backendKey: string;  // e.g., "vm"
}

// Date Range Preset Options
export type DateRangePreset = 'today' | 'yesterday' | 'last_week' | 'last_month' | 'last_6_months' | 'last_year' | 'custom';

export interface DateRangeOption {
    value: DateRangePreset;
    label: string;
}

export const DATE_RANGE_OPTIONS: DateRangeOption[] = [
    { value: 'today', label: 'Today' },
    { value: 'yesterday', label: 'Yesterday' },
    { value: 'last_week', label: 'Last Week' },
    { value: 'last_month', label: 'Last Month' },
    { value: 'last_6_months', label: 'Last 6 Months' },
    { value: 'last_year', label: 'Last Year' },
    { value: 'custom', label: 'Custom Range' },
];

// Filter State Interface
export interface RecommendationFilters {
    resourceType: string; // The selected display name (e.g., "VM")
    resourceId?: string;
    resourceIdEnabled: boolean; // New: toggle for resource ID filter
    dateRangePreset: DateRangePreset; // New: selected preset
    startDate?: Date;
    endDate?: Date;
}

// Resource Type definitions for Azure
export const AZURE_RESOURCES: CloudResourceMap[] = [
    { displayName: "VM", backendKey: "vm" },
    { displayName: "Storage Account", backendKey: "storage" },
    { displayName: "Public IP", backendKey: "publicip" },
    { displayName: "Databricks", backendKey: "databricks" },
];

// Resource Type definitions for AWS
export const AWS_RESOURCES: CloudResourceMap[] = [
    { displayName: "EC2", backendKey: "ec2" },
    { displayName: "S3", backendKey: "s3" },
];

// Resource Type definitions for GCP
export const GCP_RESOURCES: CloudResourceMap[] = [
    { displayName: "Compute Engine", backendKey: "compute" },
    { displayName: "Cloud Storage", backendKey: "storage" },
    // Add other GCP resources as needed
];