"use client";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import { useParams } from "next/navigation";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";
import ProjectBudgetInfo from "@/components/dashboard/charts/ProjectBudgetInfo";
import TagsBudgetInfo from "@/components/dashboard/charts/TagsBudgetInfo";
import TagDateFilterBar from "@/components/filters/TagDateFilterBar";
import { useState, useEffect, useCallback } from "react";
import Loader from "@/components/Loader";

const Dashboard: React.FC = () => {
  const params = useParams();
  const projectId = params.projectName as string;

  const [selectedTagId, setSelectedTagId] = useState<number | undefined>(undefined);
  const [selectedDuration, setSelectedDuration] = useState<string | undefined>(undefined);
  // Global loading state
  const [globalLoading, setGlobalLoading] = useState(false);

  // Helper to track how many children are loading
  const [loadingCount, setLoadingCount] = useState(0);
  const setChildLoading = useCallback((isLoading: boolean) => {
    setLoadingCount((prev) => {
      const next = isLoading ? prev + 1 : Math.max(prev - 1, 0);
      return next;
    });
  }, []);

  useEffect(() => {
    setGlobalLoading(loadingCount > 0);
  }, [loadingCount]);

  // Set tag ID from URL if present
  useEffect(() => {
    const tagIdString = Array.isArray(params.tag_id) ? params.tag_id[0] : params.tag_id;
    const tagId = tagIdString ? parseInt(tagIdString, 10) : undefined;
    setSelectedTagId(tagId);
  }, [params.tag_id]);

  return (
    <div className="bg-cp-card shadow-md rounded-md border border-cp-border w-full h-full overflow-y-auto p-4 space-y-4">
      <div className="mb-4">
        <div className="flex flex-col md:flex-row mb-4 md:space-x-4 space-y-2 md:space-y-0">
          <div className="w-full">
            <TagDateFilterBar onTagSelect={setSelectedTagId} onDateRangeSelect={setSelectedDuration} />
          </div>
        </div>
      </div>

      {/* Main dashboard content with loader overlay */}
      <div className="relative">
        {globalLoading && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-white bg-opacity-60">
            <Loader />
          </div>
        )}
        <div className="mb-4">
          {selectedTagId ? (
            <TagsBudgetInfo
              cloudPlatform="azure"
              queryType="tags_azure_budgets_data"
              projectId={projectId}
              title="Azure Budget"
              tag_id={selectedTagId}
              // duration={selectedDuration}
              setParentLoading={setChildLoading}
            />
          ) : (
            <ProjectBudgetInfo
              cloudPlatform="azure"
              queryType="azure_budgets_data"
              projectId={projectId}
              title="Azure Budget"
              // duration={selectedDuration}
              setParentLoading={setChildLoading}
            />
          )}
        </div>

        {/* Charts can be updated similarly if they use duration */}
        <div className="h-[400px] items-center justify-center flex">
          <LineChartComponent
            title="COST OVER TIME"
            cloudPlatform="azure"
            queryType={
              selectedTagId
                ? "azure_cost_trends_over_time_tags"
                : "azure_cost_trends_over_time"
            }
            projectId={projectId}
            xKey="azure_fact_cost.charge_period_start"
            yKey="azure_fact_cost.total_billed_cost"
            name="Azure Cost"
            xAxisLabelFormatter={(label: string) =>
              label.length > 6 ? `${label.slice(0, 6)}...` : label
            }
            xAxisTooltipFormatter={(label: string) => label}
            tag_id={selectedTagId}
            duration={selectedDuration}
            setParentLoading={setChildLoading}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="h-[300px] items-center justify-center flex">
            <PieChartComponent
              title="COST BY RESOURCES"
              cloudPlatform="azure"
              queryType={
                selectedTagId
                  ? "azure_resource_name_cost_tags"
                  : "azure_resource_name_cost"
              }
              projectId={projectId}
              nameField="azure_resource_dim.resource_name"
              valueField="azure_fact_cost.total_billed_cost"
              tag_id={selectedTagId}
              duration={selectedDuration}
              setParentLoading={setChildLoading}
            />
          </div>
          <div className="h-[300px] items-center justify-center flex">
            <PieChartComponent
              title="COST BY SERVICE NAMES"
              cloudPlatform="azure"
              queryType={
                selectedTagId
                  ? "azure_service_name_cost_tags"
                  : "azure_service_name_cost"
              }
              projectId={projectId}
              nameField="azure_resource_dim.service_name"
              valueField="azure_fact_cost.total_billed_cost"
              tag_id={selectedTagId}
              duration={selectedDuration}
              setParentLoading={setChildLoading}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
          <div className="h-[400px] items-center justify-center flex">
            <BarChartComponent
              title="COST BY REGIONS"
              cloudPlatform="azure"
              queryType={
                selectedTagId
                  ? "azure_tags_cost_by_region"
                  : "azure_cost_by_region"
              }
              projectId={projectId}
              nameField="azure_resource_dim.region_name"
              valueField="azure_fact_cost.total_billed_cost"
              tag_id={selectedTagId}
              duration={selectedDuration}
              setParentLoading={setChildLoading}
            />
          </div>
          <div className="h-[400px] items-center justify-center flex">
            <BarChartComponent
              title="COST BY SERVICE CATEGORIES"
              cloudPlatform="azure"
              queryType={
                selectedTagId
                  ? "azure_tags_cost_by_service_category"
                  : "azure_cost_by_service_category"
              }
              projectId={projectId}
              nameField="azure_resource_dim.service_category"
              valueField="azure_fact_cost.total_billed_cost"
              tag_id={selectedTagId}
              duration={selectedDuration}
              setParentLoading={setChildLoading}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
