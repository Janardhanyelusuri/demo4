"use client";
import AreaChartComponent from "@/components/dashboard/charts/AreaChartComponent";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import { useParams } from "next/navigation";
import React from "react";
import { ArrowLeft } from "lucide-react";
import TagDateFilterBar from "@/components/filters/TagDateFilterBar";
import { useState, useEffect, useCallback } from "react";
import Loader from "@/components/Loader";

type Props = {};

const Page = (props: Props) => {
  const params = useParams();
  const projectId = params.projectName as string;
  const encodedServiceNames = params.service_name as string;
  const service_names = encodedServiceNames ? decodeURIComponent(encodedServiceNames).replace(/%20/g, " ") : undefined;

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

  // Check if tag_id exists in URL params (for direct navigation)
  useEffect(() => {
    const tagIdString = Array.isArray(params.tag_id) ? params.tag_id[0] : params.tag_id;
    const tagId = tagIdString ? parseInt(tagIdString, 10) : undefined;
    setSelectedTagId(tagId);
  }, [params.tag_id]);

  useEffect(() => {
    console.log("Selected duration:", selectedDuration);
  }, [selectedDuration]);

  console.log("Selected Tag ID:", selectedTagId);
  const costComponents = [
    {
      cloudPlatform: "azure",
      queryType: "azure_services_month_to_date_cost",
      costField: "azure_fact_cost.month_to_date_cost",
      label: `${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} Month To Date Cost`,
    },
    {
      cloudPlatform: "azure",
      queryType: "azure_services_quarter_to_date_cost",
      costField: "azure_fact_cost.quarter_to_date_cost",
      label: `${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} Quarter To Date Cost`,
    },
    {
      cloudPlatform: "azure",
      queryType: "azure_services_year_to_date_cost",
      costField: "azure_fact_cost.year_to_date_cost",
      label: `${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} Year To Date Cost`,
    },
  ];
  
  return (
    <div>
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
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {costComponents.map((component, index) => (
              <CloudCostComponent
                key={index}
                cloudPlatform={component.cloudPlatform}
                queryType={component.queryType}
                costField={component.costField}
                label={component.label}
                projectId={projectId}
                service_names={service_names}
                tag_id={selectedTagId}
                // duration={selectedDuration}
              />
            ))}
          </div>
          <div className="h-[400px] items-center justify-center flex">
            <LineChartComponent
              title={`${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} COST OVER TIME`}
              cloudPlatform="azure"
              queryType="azure_services_cost_by_time"
              projectId={projectId}
              service_names={service_names}
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
          <div className="h-[400px] items-center justify-center flex mb-4">
            <BarChartComponent
              title={`${service_names ? service_names.toUpperCase() : "UNKNOWN SERVICE"} COST BY REGIONS`}
              cloudPlatform="azure"
              queryType="azure_services_cost_by_region"
              projectId={projectId}
              service_names={service_names}
              nameField="azure_resource_dim.region_name"
              valueField="azure_fact_cost.total_billed_cost"
              tag_id={selectedTagId}
              duration={selectedDuration}
              setParentLoading={setChildLoading}
            />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="h-[400px] items-center justify-center flex">
              <PieChartComponent
                title={`${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} COST BY INSTANCES`}
                cloudPlatform="azure"
                queryType="azure_services_cost_by_instance"
                projectId={projectId}
                service_names={service_names}
                nameField="azure_resource_dim.resource_name"
                valueField="azure_fact_cost.total_billed_cost"
                tag_id={selectedTagId}
                duration={selectedDuration}
                setParentLoading={setChildLoading}
              />
            </div>
            <div className="h-[400px] items-center justify-center flex">
              <PieChartComponent
                title={`${service_names ? service_names.toUpperCase() : "UNKNOWN SERVICE"} COST BY SKU METER NAME`}
                cloudPlatform="azure"
                queryType="azure_services_cost_by_sku"
                projectId={projectId}
                service_names={service_names}
                nameField="azure_fact_cost.sku_meter_name"
                valueField="azure_fact_cost.total_billed_cost"
                tag_id={selectedTagId}
                duration={selectedDuration}
                setParentLoading={setChildLoading}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Page;

// import React from "react";

// type Props = {};

// const page = (props: Props) => {
//   return (
//     <div className="">
//       <div className="min-h-screen flex items-center justify-center">
//         <div className="max-w-2xl w-full px-4">
//           <h1 className="text-4xl font-bold text-center mb-8 text-gray-900 dark:text-white">
//             Coming Soon!
//           </h1>
//           <p className="text-lg text-gray-600 dark:text-gray-300 text-center mb-12">
//             Work in Progress
//           </p>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default page;
