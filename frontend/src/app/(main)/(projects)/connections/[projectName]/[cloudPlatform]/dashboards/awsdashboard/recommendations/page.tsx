// src/app/(main)/(projects)/connections/[projectName]/[cloudPlatform]/dashboards/awsdashboard/recommendations/page.tsx

"use client";

import React, { useState } from "react";
import { useParams } from "next/navigation";
import { NormalizedRecommendation, RecommendationFilters, AWS_RESOURCES } from "@/types/recommendations";
import { fetchRecommendationsWithFilters } from "@/lib/recommendations";

// SHARED COMPONENT IMPORTS
import RecommendationFilterBar from "@/components/recommendations/RecommendationFilterBar";
import RecommendationList from "@/components/recommendations/RecommendationList";

const AwsRecommendationsPage: React.FC = () => {
  const params = useParams();
  const projectId = params.projectName as string;
  const cloudPlatform = 'aws' as const; // Hardcoded for AWS page

  const [recommendations, setRecommendations] = useState<NormalizedRecommendation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resourceOptions = AWS_RESOURCES;

  // Initialize filters with the first AWS resource type
  const [filters, setFilters] = useState<RecommendationFilters>({
    resourceType: resourceOptions[0]?.displayName || '',
    resourceId: '',
    resourceIdEnabled: false,
    startDate: undefined,
    endDate: undefined,
    dateRangePreset: 'last_week'
  });

  const handleFetch = async () => {
    // Validation ensures analysis only runs if a resource type is selected.
    if (!filters.resourceType) {
      setError("Please select a Resource Type to analyze.");
      setRecommendations([]);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Use AWS_RESOURCES and 'aws' platform
      const result = await fetchRecommendationsWithFilters(
        projectId,
        cloudPlatform,
        filters
      );
      setRecommendations(result.recommendations);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unknown error occurred while fetching recommendations.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFilters({
      resourceType: resourceOptions[0]?.displayName || '',
      resourceId: '',
      resourceIdEnabled: false,
      startDate: undefined,
      endDate: undefined,
      dateRangePreset: 'last_week'
    });
    setRecommendations([]);
    setError(null);
  };

  // FIX: Removed the useEffect hook that caused the default load on mount.
  // The user must now click "Run Analysis" to fetch data.

  return (
    <div className="p-8">
      <h1 className="text-cp-title-2xl font-cp-semibold mb-6 text-cp-blue">
        AWS Cost Optimization Recommendations
      </h1>

      {/* FILTER BAR UI (Uses shared component with date/button validation) */}
      <RecommendationFilterBar
        filters={filters}
        setFilters={setFilters}
        resourceOptions={resourceOptions}
        isLoading={isLoading}
        onRunAnalysis={handleFetch}
        onReset={handleReset}
        projectId={projectId}
        cloudPlatform={cloudPlatform}
      />

      {/* RESULTS DISPLAY */}
      {isLoading ? (
        <div className="p-8 text-center text-lg">Analyzing {filters.resourceType} data...</div>
      ) : error ? (
        <div className="p-8 text-center text-red-600 font-medium">Error: {error}</div>
      ) : recommendations.length === 0 ? (
        <div className="p-8 text-center bg-gray-50 border rounded-lg shadow-sm">
          <p className="text-cp-body text-gray-700">No optimization opportunities found for the selected filters.</p>
        </div>
      ) : (
        <RecommendationList recommendations={recommendations} />
      )}
    </div>
  );
};

export default AwsRecommendationsPage;