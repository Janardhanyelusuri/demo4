// src/lib/recommendations.ts

import axiosInstance, { BACKEND } from "@/lib/api"; 
import { 
    RawRecommendation, 
    NormalizedRecommendation, 
    RecommendationFilters, 
    AZURE_RESOURCES, 
    AWS_RESOURCES, 
    GCP_RESOURCES,
    CloudResourceMap // REQUIRED for explicit typing
} from "@/types/recommendations";
import { format } from "date-fns"; 

// Helper function to normalize the data - preserving ALL data sections
const normalizeRecommendations = (data: RawRecommendation[]): NormalizedRecommendation[] => {
  return data.map((item) => {
    // Calculate total saving percentage
    let totalSavingPct = item.recommendations.effective_recommendation.saving_pct;
    item.recommendations.additional_recommendation.forEach(detail => {
        totalSavingPct += detail.saving_pct;
    });

    const getSeverity = (saving: number): 'High' | 'Medium' | 'Low' => {
      if (saving >= 25) return 'High';
      if (saving >= 15) return 'Medium';
      return 'Low';
    };

    return {
      // Resource info
      resourceId: item.resource_id,

      // Recommendations - preserve structure
      effectiveRecommendation: item.recommendations.effective_recommendation,
      additionalRecommendations: item.recommendations.additional_recommendation,
      baseOfRecommendations: item.recommendations.base_of_recommendations,
      totalSavingPercent: parseFloat(totalSavingPct.toFixed(2)),

      // Forecasting - preserve full object
      costForecasting: item.cost_forecasting,

      // Anomalies - preserve full array
      anomalies: item.anomalies,

      // Contract/Deal - preserve full object
      contractDeal: item.contract_deal,

      // Severity for UI styling
      severity: getSeverity(totalSavingPct),
    } as NormalizedRecommendation;
  });
};

/**
 * Helper to map frontend display name to the backend resource key.
 */
const getBackendKey = (cloud: string, displayName: string): string | undefined => {
    // ⭐ FIX APPLIED: Explicitly type the resources array to resolve the visual error
    let resources: CloudResourceMap[] = []; 
    
    if (cloud === 'azure') resources = AZURE_RESOURCES;
    else if (cloud === 'aws') resources = AWS_RESOURCES;
    else if (cloud === 'gcp') resources = GCP_RESOURCES;

    const map = resources.find(r => r.displayName === displayName);
    return map?.backendKey;
};


/**
 * Public function: Fetches recommendations based on user-selected filters.
 * @param signal - Optional AbortSignal to cancel the request
 * @returns Object containing recommendations array and optional task_id
 */
export const fetchRecommendationsWithFilters = async (
    projectId: string,
    cloudPlatform: 'azure' | 'aws' | 'gcp',
    filters: RecommendationFilters,
    signal?: AbortSignal
): Promise<{ recommendations: NormalizedRecommendation[], taskId?: string }> => {
    
    // 1. Get the internal backend key from the selected display name
    const backendKey = getBackendKey(cloudPlatform, filters.resourceType);

    if (!backendKey) {
        throw new Error("Invalid resource type selected.");
    }

    const url = `${BACKEND}/llm/${cloudPlatform}/${projectId}`; 
    
    // 2. Prepare the payload, formatting dates to ISO string if they exist
    const body = {
        resource_type: backendKey,
        resource_id: filters.resourceId || undefined,
        // Format dates as ISO strings (FastAPI standard)
        start_date: filters.startDate ? format(filters.startDate, "yyyy-MM-dd'T'HH:mm:ss") : undefined,
        end_date: filters.endDate ? format(filters.endDate, "yyyy-MM-dd'T'HH:mm:ss") : undefined,
    };

    try {
        const response = await axiosInstance.post(url, body, {
            headers: { "Content-Type": "application/json" },
            signal: signal // Pass the abort signal to axios
        });

        // Extract task_id from response header (available immediately) or body (fallback)
        const taskId = response.headers['x-task-id'] || response.data.task_id;

        // 3. Parse the JSON string from the 'recommendations' field
        const rawJsonString = response.data.recommendations;

        if (rawJsonString) {
             const rawData = JSON.parse(rawJsonString) as RawRecommendation[];
             // 4. Normalize and return with task_id
             return {
                 recommendations: normalizeRecommendations(rawData),
                 taskId: taskId
             };
        }

        return {
            recommendations: [],
            taskId: taskId
        };

    } catch (err: any) {
        // Handle request cancellation gracefully
        if (err.name === 'CanceledError' || err.name === 'AbortError') {
            console.log('Request was cancelled by user');
            throw new Error('Analysis cancelled');
        }

        console.error(`API Error fetching ${cloudPlatform} ${backendKey}:`, err);
        // Throw a user-friendly error after logging the technical details
        throw new Error(`Failed to load ${filters.resourceType} analysis. Please check the backend service.`);
    }
};