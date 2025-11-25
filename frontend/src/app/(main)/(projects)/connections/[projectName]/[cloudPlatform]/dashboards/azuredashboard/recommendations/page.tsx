// src/app/(main)/(projects)/connections/[projectName]/[cloudPlatform]/dashboards/azuredashboard/recommendations/page.tsx

"use client";

import React, { useState, useRef } from "react";
import { useParams } from "next/navigation";
import { NormalizedRecommendation, RecommendationFilters, AZURE_RESOURCES } from "@/types/recommendations";
import { fetchRecommendationsWithFilters } from "@/lib/recommendations";
import axiosInstance, { BACKEND } from "@/lib/api";

// NEW SHARED COMPONENT IMPORTS
import RecommendationFilterBar from "@/components/recommendations/RecommendationFilterBar";
import RecommendationList from "@/components/recommendations/RecommendationList";
import { Button } from "@/components/ui/button";

const AzureRecommendationsPage: React.FC = () => {
  const params = useParams();
  const projectId = params.projectName as string;
  const cloudPlatform = 'azure' as const;

  const [recommendations, setRecommendations] = useState<NormalizedRecommendation[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);

  // Generation counter: increments with each new analysis or reset
  // Only responses matching current generation are processed
  const generationRef = useRef<number>(0);
  // Store current task ID for optional backend cleanup
  const currentTaskIdRef = useRef<string | null>(null);

  const resourceOptions = AZURE_RESOURCES;

  // Initialize filters with default values including new properties
  const [filters, setFilters] = useState<RecommendationFilters>({
    resourceType: resourceOptions[0]?.displayName || '',
    resourceId: undefined,
    resourceIdEnabled: false, // New: toggle off by default
    dateRangePreset: 'last_month', // New: default to last month
    startDate: undefined,
    endDate: undefined,
  });

  // Backend cancellation using NEW non-auth endpoint for instant response
  const cancelBackendTask = async (projectIdToCancel: string) => {
    const cancelUrl = `${BACKEND}/cancel-tasks/${projectIdToCancel}`;
    console.log(`🔄 [NO-AUTH] Starting FAST cancel request to: ${cancelUrl}`);
    console.log(`🔍 [DEBUG] BACKEND constant value: ${BACKEND}`);

    // Try multiple methods to ensure the request gets through

    // Method 1: navigator.sendBeacon (most reliable for fire-and-forget)
    if (typeof navigator.sendBeacon === 'function') {
      const sent = navigator.sendBeacon(cancelUrl);
      console.log(`📡 [NO-AUTH] sendBeacon result: ${sent}`);
    }

    // Method 2: fetch as backup (with detailed logging)
    try {
      console.log(`🔄 [DEBUG] About to call fetch()...`);
      fetch(cancelUrl, {
        method: 'POST',
      }).then(response => {
        console.log(`✅ [NO-AUTH] fetch() got response: ${response.status}`);
        if (response.ok) {
          return response.json();
        }
        throw new Error(`HTTP ${response.status}`);
      }).then(data => {
        console.log(`✅ [NO-AUTH] Cancel completed: cancelled ${data.cancelled_count} task(s)`);
      }).catch(error => {
        console.error(`❌ [NO-AUTH] fetch() failed:`, error);
      });
      console.log(`⚡ [DEBUG] fetch() call initiated`);
    } catch (error) {
      console.error(`❌ [DEBUG] Exception calling fetch():`, error);
    }
  };

  const handleFetch = async () => {
    // Increment generation - this invalidates all previous requests
    generationRef.current += 1;
    const thisGeneration = generationRef.current;

    console.log(`🚀 [RESET-v4.0-NO-AUTH] Starting analysis (generation ${thisGeneration})`);

    // Validation
    if (!filters.resourceType) {
      setError("Please select a Resource Type to analyze.");
      setRecommendations([]);
      return;
    }

    if (!filters.startDate || !filters.endDate) {
      setError("Please select a date range.");
      setRecommendations([]);
      return;
    }

    // Cancel previous backend task (non-blocking with keepalive)
    if (currentTaskIdRef.current) {
      cancelBackendTask(projectId);
      currentTaskIdRef.current = null;
    }

    setIsLoading(true);
    setError(null);
    setCurrentIndex(0);

    try {
      const result = await fetchRecommendationsWithFilters(
        projectId,
        cloudPlatform,
        filters,
        undefined // No abort signal needed
      );

      // CRITICAL: Only process if this is still the current generation
      if (generationRef.current !== thisGeneration) {
        console.log(`⚠️  Ignoring response from old generation ${thisGeneration} (current: ${generationRef.current})`);
        return;
      }

      // Store task_id for cleanup
      if (result.taskId) {
        currentTaskIdRef.current = result.taskId;
        console.log(`📋 Task started: ${result.taskId}`);
      }

      setRecommendations(result.recommendations);
      console.log(`✅ Analysis complete (generation ${thisGeneration}): ${result.recommendations.length} recommendations`);
    } catch (err) {
      // Only show errors for current generation
      if (generationRef.current !== thisGeneration) {
        console.log(`⚠️  Ignoring error from old generation ${thisGeneration}`);
        return;
      }

      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unknown error occurred while fetching recommendations.");
      }
    } finally {
      // Only clear loading if still current generation
      if (generationRef.current === thisGeneration) {
        setIsLoading(false);
      }
    }
  };

  // Reset: Increment generation + AWAIT backend cancel before clearing UI
  const handleReset = async () => {
    // Increment generation - this makes all in-flight requests obsolete
    generationRef.current += 1;

    // IMMEDIATELY stop loading indicator so user sees response
    setIsLoading(false);

    console.log(`🔄 [RESET-v4.0-NO-AUTH] Reset clicked (new generation: ${generationRef.current})`);

    // CRITICAL: AWAIT the cancel request to ensure it completes before state updates
    if (currentTaskIdRef.current || projectId) {
      await cancelBackendTask(projectId);  // Wait for it to complete!
      currentTaskIdRef.current = null;
      console.log(`✅ [DEBUG] Cancel request completed, now clearing UI...`);
    }

    // Clear UI AFTER cancel request completes
    setFilters({
      resourceType: resourceOptions[0]?.displayName || '',
      resourceId: undefined,
      resourceIdEnabled: false,
      dateRangePreset: 'last_month',
      startDate: undefined,
      endDate: undefined,
    });

    setRecommendations([]);
    setCurrentIndex(0);
    setError(null);
    setIsTransitioning(false);

    console.log(`✅ Reset complete - UI cleared, generation ${generationRef.current}`);
  };

  return (
    <div className="p-4">
      {/* FILTER BAR UI - Moved to top, removed title */}
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
        <div className="p-6 text-center">
          <div className="animate-pulse">
            <div className="h-6 bg-gray-200 rounded w-1/2 mx-auto mb-3"></div>
            <div className="h-3 bg-gray-200 rounded w-1/3 mx-auto"></div>
          </div>
          <p className="mt-3 text-sm text-gray-600">Analyzing {filters.resourceType} data...</p>
        </div>
      ) : error ? (
        <div className="p-5 text-center text-red-600 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-base font-semibold mb-1">Error</p>
          <p className="text-sm">{error}</p>
        </div>
      ) : recommendations.length === 0 ? (
        <div className="p-6 text-center bg-gray-50 border rounded-lg shadow-sm">
          <p className="text-sm text-gray-700">
            No optimization opportunities found for the selected filters.
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Try adjusting your date range or resource selection.
          </p>
        </div>
      ) : (
        <RecommendationList recommendations={recommendations} />
      )}
    </div>
  );
};

export default AzureRecommendationsPage;
