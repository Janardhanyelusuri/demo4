"use client";
import AreaChartComponent from "@/components/dashboard/charts/AreaChartComponent";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import { useParams } from "next/navigation";
import React from "react";

type Props = {};

const Page = (props: Props) => {
  const params = useParams();
  const projectId = params.projectName as string;
  const encodedServiceNames = params.service_name as string;
  const service_names = encodedServiceNames ? decodeURIComponent(encodedServiceNames).replace(/%20/g, " ") : undefined;

  const costComponents = [
    {
      cloudPlatform: "aws",
      queryType: "aws_services_month_to_date_cost",
      costField: "aws_fact_focus.month_to_date_cost",
      label: `${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} Month To Date Cost`,
    },
    {
      cloudPlatform: "aws",
      queryType: "aws_services_quarter_to_date_cost",
      costField: "aws_fact_focus.quarter_to_date_cost",
      label: `${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} Quarter To Date Cost`,
    },
    {
      cloudPlatform: "aws",
      queryType: "aws_services_year_to_date_cost",
      costField: "aws_fact_focus.year_to_date_cost",
      label: `${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} Year To Date Cost`,
    },
  ];
  
  return (
    <div>
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
              service_names={service_names}
            />
          ))}
        </div>
        <div className="h-[400px] items-center justify-center flex">
          <LineChartComponent
            title={`${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} COST OVER TIME`}
            cloudPlatform="aws"
            queryType="aws_services_cost_by_time"
            projectId={projectId}
            service_names={service_names}
            xKey="aws_fact_focus.charge_period_start"
            yKey="aws_fact_focus.total_list_cost"
            name="AWS Cost"
            xAxisLabelFormatter={(label: string) =>
              label.length > 6 ? `${label.slice(0, 6)}...` : label
            }
            xAxisTooltipFormatter={(label: string) => label}
          />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="h-[400px] items-center justify-center flex mb-4">
            <BarChartComponent
              title={`${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} COST BY REGIONS`}
              cloudPlatform="aws"
              queryType="aws_services_cost_by_region"
              projectId={projectId}
              service_names={service_names}
              nameField="aws_fact_focus.region_name"
              valueField="aws_fact_focus.total_list_cost"
            />
          </div>
          <div className="h-[400px] items-center justify-center flex">
            <PieChartComponent
              title={`${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} COST BY INSTANCES`}
              cloudPlatform="aws"
              queryType="aws_services_cost_by_instance"
              projectId={projectId}
              service_names={service_names}
              nameField="aws_fact_focus.resource_id"
              valueField="aws_fact_focus.total_list_cost"
            />
          </div>
          {/* <div className="h-[400px] items-center justify-center flex">
          <PieChartComponent
            title={`${service_names ? service_names.toUpperCase() : 'UNKNOWN SERVICE'} COST BY SKU METER NAME`}
            cloudPlatform="azure"
            queryType="azure_services_cost_by_sku"
            projectId={projectId}
            service_names={service_names}
            nameField="azure_fact_cost.sku_meter_name"
            valueField="azure_fact_cost.total_billed_cost"
          />
        </div> */}
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
