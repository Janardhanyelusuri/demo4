// src/components/recommendations/RecommendationFilterBar.tsx

import React, { useState, useEffect, useRef } from 'react';
import { RecommendationFilters, CloudResourceMap, DATE_RANGE_OPTIONS, DateRangePreset } from "@/types/recommendations";
import { format } from "date-fns";
import { cn } from "@/lib/utils";
import { CalendarIcon, ChevronDown, RotateCw } from "lucide-react";
import { calculateDateRange } from "@/lib/dateUtils";
import axiosInstance, { BACKEND } from "@/lib/api";

// --- UI Imports (Shadcn/Radix components) ---
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface ResourceIdOption {
    resource_id: string;
    resource_name: string;
}

interface RecommendationFilterBarProps {
    filters: RecommendationFilters;
    setFilters: React.Dispatch<React.SetStateAction<RecommendationFilters>>;
    resourceOptions: CloudResourceMap[];
    isLoading: boolean;
    onRunAnalysis: () => void;
    onReset: () => void;
    projectId: string;
    cloudPlatform: 'azure' | 'aws' | 'gcp';
}

const RecommendationFilterBar: React.FC<RecommendationFilterBarProps> = ({
    filters,
    setFilters,
    resourceOptions,
    isLoading,
    onRunAnalysis,
    onReset,
    projectId,
    cloudPlatform
}) => {
    const [resourceIds, setResourceIds] = useState<ResourceIdOption[]>([]);
    const [loadingResourceIds, setLoadingResourceIds] = useState(false);
    const abortControllerRef = useRef<AbortController | null>(null);

    // Define the date boundaries to prevent future date selection
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Fetch resource IDs when resource type changes (with debounce)
    useEffect(() => {
        if (!filters.resourceType || !filters.resourceIdEnabled) {
            setResourceIds([]);
            setLoadingResourceIds(false);
            return;
        }

        // Debounce: wait 300ms before fetching to prevent rapid changes
        const debounceTimer = setTimeout(() => {
            // Cancel any previous request
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }

            // Create new AbortController for this request
            abortControllerRef.current = new AbortController();
            let isMounted = true;

            const fetchResourceIds = async () => {
                setLoadingResourceIds(true);
                setResourceIds([]); // Clear existing data before fetching

                try {
                    const resourceMap = resourceOptions.find(r => r.displayName === filters.resourceType);
                    if (!resourceMap) {
                        if (isMounted) {
                            setLoadingResourceIds(false);
                        }
                        return;
                    }

                    const url = `${BACKEND}/llm/${cloudPlatform}/${projectId}/resources/${resourceMap.backendKey}`;

                    const response = await axiosInstance.get(url, {
                        signal: abortControllerRef.current!.signal
                    });

                    if (isMounted && response.data.status === 'success') {
                        const ids = response.data.resource_ids || [];
                        setResourceIds(ids);
                    }
                } catch (error: any) {
                    // Clear resourceIds on error to ensure clean state
                    if (isMounted) {
                        setResourceIds([]);
                    }

                    // Don't log error if request was cancelled
                    if (error.name !== 'CanceledError' && error.name !== 'AbortError') {
                        console.error('Error fetching resource IDs:', error);
                    }
                } finally {
                    if (isMounted) {
                        setLoadingResourceIds(false);
                        abortControllerRef.current = null;
                    }
                }
            };

            fetchResourceIds();

            // Cleanup function
            return () => {
                isMounted = false;
            };
        }, 300);

        // Cleanup: clear debounce timer and abort request
        return () => {
            clearTimeout(debounceTimer);
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
                abortControllerRef.current = null;
            }
        };
    }, [filters.resourceType, filters.resourceIdEnabled, projectId, cloudPlatform]);

    // Handle date range preset change
    const handleDateRangePresetChange = (preset: DateRangePreset) => {
        setFilters(prev => {
            const dateRange = calculateDateRange(preset);

            return {
                ...prev,
                dateRangePreset: preset,
                startDate: dateRange?.startDate,
                endDate: dateRange?.endDate
            };
        });
    };

    // Handle resource ID toggle
    const handleResourceIdToggle = (enabled: boolean) => {
        setFilters(prev => ({
            ...prev,
            resourceIdEnabled: enabled,
            resourceId: enabled ? prev.resourceId : undefined
        }));
    };

    return (
        <div className="bg-gradient-to-br from-[#F9FEFF] to-[#EAF1FB] rounded-xl shadow-lg border border-[#233E7D]/30 mb-4 p-4">
            {/* First Row: Main filters with Run Analysis button at the end */}
            <div className="flex flex-wrap items-center gap-2">
                {/* Resource Type */}
                <div className="flex-shrink-0" style={{ minWidth: '160px' }}>
                    <Select
                        value={filters.resourceType}
                        onValueChange={(value) => setFilters(prev => ({
                            ...prev,
                            resourceType: value,
                            resourceId: undefined
                        }))}
                    >
                        <SelectTrigger className="h-9 bg-white text-[#233E7D] text-xs font-semibold border-[#B6C6E3] focus:ring-2 focus:ring-[#233E7D]/40 hover:bg-[#EAF1FB] transition-colors">
                            <SelectValue placeholder="Resource Type" />
                        </SelectTrigger>
                        <SelectContent>
                            {resourceOptions.map((r) => (
                                <SelectItem key={r.backendKey} value={r.displayName}>
                                    {r.displayName}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                {/* Resource ID Toggle */}
                <div className="flex items-center space-x-2 px-3 py-1.5 bg-white rounded-lg border border-[#233E7D]/20 flex-shrink-0">
                    <Switch
                        id="resource-id-toggle"
                        checked={filters.resourceIdEnabled}
                        onCheckedChange={handleResourceIdToggle}
                        className="data-[state=checked]:bg-[#233E7D]"
                    />
                    <Label htmlFor="resource-id-toggle" className="text-xs font-medium text-[#233E7D] cursor-pointer whitespace-nowrap">
                        Specific Resource
                    </Label>
                </div>

                {/* Resource ID Dropdown - Only show when enabled */}
                {filters.resourceIdEnabled && (
                    <div className="flex-shrink-0" style={{ minWidth: '200px' }}>
                        <Select
                            value={filters.resourceId || ""}
                            onValueChange={(value) => setFilters(prev => ({ ...prev, resourceId: value }))}
                            disabled={loadingResourceIds || !filters.resourceType}
                        >
                            <SelectTrigger className="h-9 bg-white text-[#233E7D] text-xs font-semibold border-[#B6C6E3] focus:ring-2 focus:ring-[#233E7D]/40 hover:bg-[#EAF1FB] transition-colors">
                                <SelectValue placeholder={loadingResourceIds ? "Loading..." : "Select Resource"} />
                            </SelectTrigger>
                            <SelectContent className="max-h-[300px]">
                                {resourceIds.length === 0 ? (
                                    <div className="p-2 text-xs text-gray-500">No resources found</div>
                                ) : (
                                    resourceIds.map((resource) => {
                                        // Show last 3 parts of resource ID for better context
                                        const parts = resource.resource_id.split('/');
                                        const resourceIdShort = parts.length > 3
                                            ? parts.slice(-3).join('/')
                                            : resource.resource_id;

                                        return (
                                            <SelectItem
                                                key={resource.resource_id}
                                                value={resource.resource_id}
                                                title={resource.resource_id}
                                            >
                                                <div className="flex flex-col">
                                                    <span className="font-medium text-xs">{resourceIdShort}</span>
                                                    <span className="text-[10px] text-gray-500 truncate max-w-md">
                                                        {resource.resource_name}
                                                    </span>
                                                </div>
                                            </SelectItem>
                                        );
                                    })
                                )}
                            </SelectContent>
                        </Select>
                    </div>
                )}

                {/* Date Range Preset */}
                <div className="flex-shrink-0" style={{ minWidth: '140px' }}>
                    <Select
                        value={filters.dateRangePreset}
                        onValueChange={(value) => handleDateRangePresetChange(value as DateRangePreset)}
                    >
                        <SelectTrigger className="h-9 bg-white text-[#233E7D] text-xs font-semibold border-[#B6C6E3] focus:ring-2 focus:ring-[#233E7D]/40 hover:bg-[#EAF1FB] transition-colors">
                            <SelectValue placeholder="Date Range" />
                        </SelectTrigger>
                        <SelectContent>
                            {DATE_RANGE_OPTIONS.map((option) => (
                                <SelectItem key={option.value} value={option.value}>
                                    {option.label}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                {/* Display selected date range for non-custom presets */}
                {filters.dateRangePreset !== 'custom' && filters.startDate && filters.endDate && (
                    <div className="flex-shrink-0">
                        <div className="text-xs text-[#233E7D] bg-white px-3 py-2 rounded-md border border-[#233E7D]/20 font-medium whitespace-nowrap h-9 flex items-center">
                            {format(filters.startDate, "MMM d")} - {format(filters.endDate, "MMM d, yyyy")}
                        </div>
                    </div>
                )}

                {/* Reset Button and Run Analysis Button - Stick to the end */}
                <div className="flex items-center gap-2 flex-shrink-0 ml-auto">
                    {/* Reset Button */}
                    <Button
                        onClick={onReset}
                        variant="outline"
                        className="h-9 w-9 p-0 border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D] hover:text-white transition-all duration-300"
                        title="Reset all filters and clear results"
                    >
                        <RotateCw className="h-4 w-4" />
                    </Button>

                    {/* Run Analysis Button */}
                    <Button
                        onClick={onRunAnalysis}
                        disabled={isLoading || !filters.resourceType || !filters.startDate || !filters.endDate}
                        className="h-9 px-6 text-xs font-bold bg-gradient-to-r from-[#233E7D] to-[#1a2d5c] hover:from-[#1a2d5c] hover:to-[#0f1a3a] text-white shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-105 whitespace-nowrap"
                    >
                        {isLoading ? (
                            <span className="flex items-center gap-2">
                                <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                </svg>
                                Analyzing...
                            </span>
                        ) : 'Run Analysis'}
                    </Button>
                </div>
            </div>

            {/* Second Row: Custom Date Pickers - Only show when Custom is selected */}
            {filters.dateRangePreset === 'custom' && (
                <div className="flex flex-wrap items-center gap-2 mt-3 pt-3 border-t border-[#233E7D]/10">
                    <div className="flex-shrink-0" style={{ minWidth: '140px' }}>
                        <Popover>
                            <PopoverTrigger asChild>
                                <Button
                                    variant="outline"
                                    className={cn(
                                        "h-9 w-full justify-start text-xs bg-white text-[#233E7D] font-semibold border-[#B6C6E3] hover:bg-[#EAF1FB] transition-colors",
                                        !filters.startDate && "text-muted-foreground"
                                    )}
                                >
                                    <CalendarIcon className="mr-1.5 h-3 w-3" />
                                    {filters.startDate ? format(filters.startDate, "MMM d, yy") : "Start Date"}
                                </Button>
                            </PopoverTrigger>
                            <PopoverContent className="w-auto p-0" align="start">
                                <Calendar
                                    mode="single"
                                    selected={filters.startDate}
                                    onSelect={(date) => setFilters(prev => ({ ...prev, startDate: date }))}
                                    initialFocus
                                    disabled={(date) => date > today}
                                />
                            </PopoverContent>
                        </Popover>
                    </div>

                    <div className="flex-shrink-0" style={{ minWidth: '140px' }}>
                        <Popover>
                            <PopoverTrigger asChild>
                                <Button
                                    variant="outline"
                                    className={cn(
                                        "h-9 w-full justify-start text-xs bg-white text-[#233E7D] font-semibold border-[#B6C6E3] hover:bg-[#EAF1FB] transition-colors",
                                        !filters.endDate && "text-muted-foreground"
                                    )}
                                >
                                    <CalendarIcon className="mr-1.5 h-3 w-3" />
                                    {filters.endDate ? format(filters.endDate, "MMM d, yy") : "End Date"}
                                </Button>
                            </PopoverTrigger>
                            <PopoverContent className="w-auto p-0" align="start">
                                <Calendar
                                    mode="single"
                                    selected={filters.endDate}
                                    onSelect={(date) => setFilters(prev => ({ ...prev, endDate: date }))}
                                    initialFocus
                                    disabled={(date) =>
                                       date > today ||
                                       (filters.startDate ? date < filters.startDate : false)
                                    }
                                />
                            </PopoverContent>
                        </Popover>
                    </div>
                </div>
            )}
        </div>
    );
};

export default RecommendationFilterBar;
