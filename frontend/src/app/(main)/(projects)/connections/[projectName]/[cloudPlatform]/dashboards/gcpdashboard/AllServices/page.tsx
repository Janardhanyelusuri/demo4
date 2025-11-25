"use client";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import { useParams } from "next/navigation";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import TableComponent from "@/components/dashboard/charts/TableComponent";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";
import ProjectBudgetInfo from "@/components/dashboard/charts/ProjectBudgetInfo";

const Dashboard: React.FC = () => {
  const params = useParams();
  const projectId = params.projectName as string;
  return (
    <div className="bg-white shadow-md rounded-md border w-full h-full overflow-y-auto p-4 space-y-4">
      {/* Table Component */}
      <div className="mb-4">
        <TableComponent
          cloudPlatform="gcp"
          queryType="gcp_services_billed_cost"
          projectId={projectId}
          title="GCP Services Billed Cost"
        />
      </div>
    </div>
  );
};

export default Dashboard;
