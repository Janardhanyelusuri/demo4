"use client";

import React from "react";
import { useParams } from "next/navigation";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";

const costComponents = [
  {
    cloudPlatform: "aws",
    queryType: "kms_month_to_date_cost",
    costField: "aws_fact_cost.kms_month_to_date_cost",
    label: "kms Month to Date Cost",
  },
  {
    cloudPlatform: "aws",
    queryType: "kms_quarter_to_date_cost",
    costField: "aws_fact_cost.kms_quarter_to_date_cost",
    label: "kms Quarter to Date Cost",
  },
  {
    cloudPlatform: "aws",
    queryType: "kms_year_to_date_cost",
    costField: "aws_fact_cost.kms_year_to_date_cost",
    label: "kms Year to Date Cost",
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
          title="AWS Cost by Time Key Management Service"
          cloudPlatform="aws"
          queryType="aws_cost_by_time_kms"
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

      <div className="grid grid-cols-1 lg:grid-cols-1 gap-4">
        <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="Aws Cost by Region Key Management Service"
            cloudPlatform="aws"
            queryType="aws_cost_by_region_kms"
            projectId={projectId}
            nameField="aws_fact_focus.region_name"
            valueField="aws_fact_focus.total_list_cost"
          />
        </div>
        {/* <div className="h-[300px] items-center justify-center flex">
          <BarChartComponent
            title="Aws Cost by Instance Key Management Service"
            cloudPlatform="aws"
            queryType="aws_cost_by_instance_kms"
            projectId={projectId}
            nameField="aws_fact_focus.resource_id"
            valueField="aws_fact_focus.total_list_cost"
          />
        </div> */}
      </div>
    </div>
  );
};

export default Page;

// "use client";

// import React from "react";
// import { useParams } from "next/navigation";
// import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";

// const costComponents = [
//   {
//     cloudPlatform: "aws",
//     queryType: "aws_cost_by_region_kms",
//     costField: "aws_fact_focus.total_list_cost",
//     label: "Aws Cost by Region Kms",
//   },
// ];

// const Page: React.FC = () => {
//   const params = useParams();
//   const projectId = params.projectName as string;
//   return (
//     <div className="bg-white shadow-md rounded-md border w-full h-full overflow-y-auto p-4 space-y-4">
//       <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
//         {costComponents.map((component, index) => (
//           <CloudCostComponent
//             key={index}
//             cloudPlatform={component.cloudPlatform}
//             queryType={component.queryType}
//             costField={component.costField}
//             label={component.label}
//             projectId={projectId}
//           />
//         ))}
//       </div>
//     </div>
//   );
// };

// export default Page;
