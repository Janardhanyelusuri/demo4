"use client";
import React, { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import ProjectBudgetInfo from "@/components/dashboard/charts/ProjectBudgetInfo";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";

const costComponents = [
];

const Page: React.FC = () => {
  const params = useParams();
  const projectId = params.projectName as string;
  return (
    <div className="bg-white shadow-md rounded-md border w-full h-full overflow-y-auto p-4 space-y-4">
      <div className="mb-4">
        {" "}
        <ProjectBudgetInfo
          cloudPlatform="gcp"
          queryType="gcp_budgets_data"
          projectId={projectId as string}
          title="GCP Budget"
        />
      </div>
      <div className="h-[400px] items-center justify-center flex">
        <LineChartComponent
          title="COST OVER TIME"
          cloudPlatform="gcp"
          queryType="gcp_cost_by_time"
          projectId={projectId}
          xKey="gcp_fact_dim.charge_period_start"
          yKey="gcp_fact_dim.total_list_cost"
          name="Cost By Time"
          xAxisLabelFormatter={(label: string) =>
            label.length > 6 ? `${label.slice(0, 6)}...` : label
          }
          xAxisTooltipFormatter={(label: string) => label}
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="h-[300px] items-center justify-center flex">
          <PieChartComponent
            title="COST BY RESOURCES"
            cloudPlatform="gcp"
            queryType="gcp_resource_name_list_cost"
            projectId={projectId}
            nameField="gcp_fact_dim.resource_name"
            valueField="gcp_fact_dim.total_list_cost"
          />
        </div>
        <div className="h-[300px] items-center justify-center flex">
          <PieChartComponent
            title="COST BY SERVICE NAMES"
            cloudPlatform="gcp"
            queryType="gcp_service_name_list_cost"
            projectId={projectId}
            nameField="gcp_fact_dim.service_name"
            valueField="gcp_fact_dim.total_list_cost"
          />
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="COST BY REGIONS"
            cloudPlatform="gcp"
            queryType="gcp_cost_by_region"
            projectId={projectId}
            nameField="gcp_fact_dim.region_name"
            valueField="gcp_fact_dim.total_list_cost"
          />
        </div>
        <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="COST BY SERVICE CATEGORIES"
            cloudPlatform="gcp"
            queryType="gcp_cost_by_service_category"
            projectId={projectId}
            nameField="gcp_fact_dim.service_category"
            valueField="gcp_fact_dim.total_list_cost"
          />
        </div>
      </div>
    </div>
  );
};

export default Page;
