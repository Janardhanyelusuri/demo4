"use client";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import { useParams } from "next/navigation";
import React from "react";

type Props = {};

const costComponents = [
  {
    cloudPlatform: "azure",
    queryType: "azure_vm_month_to_date_cost",
    costField: "azure_fact_cost.vm_month_to_date_cost",
    label: "VM Month To Date Cost",
  },
  {
    cloudPlatform: "azure",
    queryType: "azure_vm_quarter_to_date_cost",
    costField: "azure_fact_cost.vm_quarter_to_date_cost",
    label: "VM Quarter To Date Costt",
  },
  {
    cloudPlatform: "azure",
    queryType: "azure_vm_year_to_date_cost",
    costField: "azure_fact_cost.vm_year_to_date_cost",
    label: "VM Year to Date Cost",
  },
];

const Page = (props: Props) => {
  const params = useParams();
  const projectId = params.projectName as string;
  return (
    <div>
      <div className="bg-cp-card shadow-md rounded-md border border-cp-border w-full h-full overflow-y-auto p-4 space-y-4">
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
            title="VM COST OVER TIME"
            cloudPlatform="azure"
            queryType="azure_cost_by_time_vm"
            projectId={projectId}
            xKey="azure_fact_cost.charge_period_start"
            yKey="azure_fact_cost.total_billed_cost"
            name="Azure Cost By Time Vm"
            xAxisLabelFormatter={(label: string) =>
              label.length > 6 ? `${label.slice(0, 6)}...` : label
            }
            xAxisTooltipFormatter={(label: string) => label}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="VM COST BY REGION"
            cloudPlatform="azure"
            queryType="azure_cost_by_region_vm"
            projectId={projectId}
            nameField="azure_resource_dim.region_name"
            valueField="azure_fact_cost.total_billed_cost"
          />
        </div>
        <div className="h-[400px] items-center justify-center flex">
          <PieChartComponent
            title="VM COST BY SKU METER NAME"
            cloudPlatform="azure"
            queryType="azure_cost_by_sku_vm"
            projectId={projectId}
            nameField="azure_fact_cost.sku_meter_name"
            valueField="azure_fact_cost.total_billed_cost"
          />
        </div>
        </div>
      </div>
    </div>
  );
};

export default Page;
