"use client";

import React from "react";
import { useParams } from "next/navigation";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";

const costComponents = [
  {
    cloudPlatform: "aws",
    queryType: "storage_month_to_date_cost",
    costField: "aws_fact_cost.storage_month_to_date_cost",
    label: "S3 Month to Date Cost",
  },
  {
    cloudPlatform: "aws",
    queryType: "storage_quarter_to_date_cost",
    costField: "aws_fact_cost.storage_quarter_to_date_cost",
    label: "S3 Quarter to Date Cost",
  },
  {
    cloudPlatform: "aws",
    queryType: "storage_year_to_date_cost",
    costField: "aws_fact_cost.storage_year_to_date_cost",
    label: "S3 Year to Date Cost",
  },
];

const Page: React.FC = () => {
  const params = useParams();
  const projectId = params.projectName as string;
  return (
    <div className="bg-white shadow-md rounded-md border w-full h-full overflow-y-auto p-4 space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {costComponents.map((component, index) => (
          <CloudCostComponent
            key={index}
            cloudPlatform={component.cloudPlatform}
            queryType={component.queryType}
            costField={component.costField}
            label={component.label}
            projectId={projectId}
          />
        ))}
      </div>
      <div className="h-[400px] items-center justify-center flex">
        <LineChartComponent
          title="AWS Cost by Time S3"
          cloudPlatform="aws"
          queryType="aws_cost_by_time_storage"
          projectId={projectId}
          xKey="aws_fact_focus.charge_period_start"
          yKey="aws_fact_focus.total_list_cost"
          name="Cost ($)"
          xAxisLabelFormatter={(label: string) =>
            label.length > 6 ? `${label.slice(0, 6)}...` : label
          }
          xAxisTooltipFormatter={(label: string) => label}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="S3 Cost by Region"
            cloudPlatform="aws"
            queryType="aws_cost_by_region_s3"
            projectId={projectId}
            nameField="aws_fact_cost.region_name"
            valueField="aws_fact_cost.total_list_cost"
          />
        </div>
        <div className="h-[400px] items-center justify-center flex">
          <PieChartComponent
            title="Aws Cost by Storageclass"
            cloudPlatform="aws"
            queryType="aws_cost_by_storageclass"
            projectId={projectId}
            nameField="aws_fact_focus.service_name"
            valueField="aws_fact_focus.total_list_cost"
          />
        </div>
      </div>
    </div>
  );
};

export default Page;
