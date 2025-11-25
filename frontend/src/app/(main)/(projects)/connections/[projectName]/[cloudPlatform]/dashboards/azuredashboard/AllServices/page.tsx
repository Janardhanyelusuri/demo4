"use client";
import React from "react";
import { useParams } from "next/navigation";
import TableComponent from "@/components/dashboard/charts/TableComponent";

const Dashboard: React.FC = () => {
  const params = useParams();
  const projectId = params.projectName as string;
  const [costStats, setCostStats] = React.useState<{ total: number; avg: number } | null>(null);

  // Handler to receive data from TableComponent
  const handleTableData = React.useCallback((tableData: any[]) => {
    if (!Array.isArray(tableData) || tableData.length === 0) {
      setCostStats({ total: 0, avg: 0 });
      return;
    }
    const costs = tableData.map((row) => Number(row.total_billed_cost) || 0);
    const total = costs.reduce((acc, val) => acc + val, 0);
    const avg = total / costs.length;
    setCostStats({ total, avg });
  }, []);

  return (
    <div className="bg-cp-card shadow-md rounded-md border border-cp-border w-full h-full overflow-y-auto p-4 space-y-4">
      {/* Table Component with cost summary above table */}
      <div className="mb-4">
      {/* Search bar and cost summary on same row */}
      <div className="flex flex-col md:flex-row items-center justify-between mb-4">
        {/* Search bar is rendered inside TableComponent, so we just align cost summary to right */}
        <div className="flex-1" />
        <div className="flex gap-4 items-center">
          <div className="bg-cp-bg px-4 py-2 rounded shadow border border-cp-border">
            <span className="font-semibold">Average Cost:</span> <span className="text-cp-primary">${costStats ? costStats.avg.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "-"}</span>
          </div>
          <div className="bg-cp-bg px-4 py-2 rounded shadow border border-cp-border">
            <span className="font-semibold">Total Cost:</span> <span className="text-cp-primary">${costStats ? costStats.total.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "-"}</span>
          </div>
        </div>
      </div>
      <TableComponent
        cloudPlatform="azure"
        queryType="azure_services_billed_cost"
        projectId={projectId}
        title="Azure Services Billed Cost"
        onDataLoaded={handleTableData}
      />
      </div>
    </div>
  );
};

export default Dashboard;
