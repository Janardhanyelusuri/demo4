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
          cloudPlatform="aws"
          queryType="aws_budgets_data"
          projectId={projectId as string}
          title="AWS Budget"
          
        />
      </div>
      <div className="h-[400px] items-center justify-center flex">
        <LineChartComponent
          title="COST OVER TIME"
          cloudPlatform="aws"
          queryType="aws_cost_by_time"
          projectId={projectId}
          xKey="aws_fact_focus.charge_period_start"
          yKey="aws_fact_focus.total_list_cost"
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
            cloudPlatform="aws"
            queryType="aws_resource_id_cost"
            projectId={projectId}
            nameField="aws_fact_focus.resource_id"
            valueField="aws_fact_focus.total_list_cost"
          />
        </div>
        <div className="h-[300px] items-center justify-center flex">
          <PieChartComponent
            title="COST BY SERVICE NAMES"
            cloudPlatform="aws"
            queryType="aws_service_name_cost"
            projectId={projectId}
            nameField="aws_fact_focus.service_name"
            valueField="aws_fact_focus.total_list_cost"
          
          />
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="COST BY REGIONS"
            cloudPlatform="aws"
            queryType="aws_cost_by_region"
            projectId={projectId}
            nameField="aws_fact_focus.region_name"
            valueField="aws_fact_focus.total_list_cost"
          />
        </div>
        <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="COST BY SERVICE CATEGORIES"
            cloudPlatform="aws"
            queryType="aws_service_category_name_cost"
            projectId={projectId}
            nameField="aws_fact_focus.service_category"
            valueField="aws_fact_focus.total_list_cost"
          />
        </div>
      </div>
    </div>
  );
};

export default Page;
