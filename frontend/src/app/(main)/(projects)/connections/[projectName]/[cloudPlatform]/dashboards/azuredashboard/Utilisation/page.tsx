"use client";

import React, { useEffect, useState, useRef } from "react";
import axiosInstance, { BACKEND } from "@/lib/api";
import { Loader2, ChevronDown, ChevronRight } from "lucide-react";
import { useParams } from "next/navigation";
import apiQueue from "@/lib/apiQueueManager";

// Import chart components
import UtilizationLineChart from "@/components/dashboard/charts/UtilizationLineChart";
import UtilizationPieChart from "@/components/dashboard/charts/UtilizationPieChart";
import UtilizationBarChart from "@/components/dashboard/charts/UtilizationBarChart";
import UtilizationMetricsTable from "@/components/dashboard/charts/UtilizationMetricsTable";


const UtilisationTable: React.FC<{ data: any[] }> = ({ data }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState<string>("");
  const [modalTitle, setModalTitle] = useState<string>("");
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const recordsPerPage = 4;
  const totalPages = Math.ceil(data.length / recordsPerPage);
  const paginatedData = data.slice((currentPage - 1) * recordsPerPage, currentPage * recordsPerPage);

  if (!data || data.length === 0) return <div>No utilisation data found.</div>;

  // Filter out 'Other' from columns for main table display
  const columns = Object.keys(data[0] || {}).filter(col => col !== 'Other');

  // Function to format column headers for better line wrapping
  const formatColumnHeader = (column: string) => {
    const words = column.split(' ');
    if (words.length <= 2) {
      return <span>{column}</span>;
    }
    const firstTwoWords = words.slice(0, 2).join(' ');
    const remainingWords = words.slice(2).join(' ');
    return (
      <div style={{ lineHeight: '1.2' }}>
        <div>{firstTwoWords}</div>
        <div>{remainingWords}</div>
      </div>
    );
  };

  const toggleRowExpansion = (rowIndex: number) => {
    setExpandedRow(expandedRow === rowIndex ? null : rowIndex);
  };

  return (
    <div className="overflow-x-auto bg-[#F9FEFF] p-4 rounded-md">
      <h2 className="text-lg font-semibold mb-4 text-[#233E7D]">Virtual Machine Recommendation</h2>
      <table className="min-w-full border border-[#E0E5EF] rounded-md bg-[#fff]">
        <thead className="bg-[#F9FEFF]">
          <tr>
            <th className="px-4 py-2 border-b border-[#E0E5EF] text-left font-medium text-sm text-[#233E7D] w-8">
              {/* Arrow column */}
            </th>
            {columns.map((col) => (
              <th 
                key={col} 
                className="px-4 py-3 border-b border-[#E0E5EF] text-left font-medium text-sm text-[#233E7D] align-top"
                style={{ minWidth: '160px', maxWidth: '240px' }}
                title={col}
              >
                {formatColumnHeader(col)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {paginatedData.map((row, idx) => {
            const actualIndex = (currentPage - 1) * recordsPerPage + idx;
            const isExpanded = expandedRow === actualIndex;
            return (
              <React.Fragment key={idx}>
                <tr className="hover:bg-[#233E7D]/5 transition-colors">
                  <td className="px-4 py-2 border-b border-[#E0E5EF] text-sm w-8">
                    {row.Other && (
                      <button
                        className="flex items-center justify-center w-6 h-6 rounded hover:bg-[#E0E5EF] transition-colors"
                        onClick={() => toggleRowExpansion(actualIndex)}
                        aria-label={isExpanded ? 'Collapse details' : 'Expand details'}
                        style={{ color: isExpanded ? '#D82026' : '#233E7D' }}
                      >
                        {isExpanded ? (
                          <ChevronDown className="w-4 h-4" />
                        ) : (
                          <ChevronRight className="w-4 h-4" />
                        )}
                      </button>
                    )}
                  </td>
                  {columns.map((col) => {
                    const value = row[col];
                    const isLong = typeof value === 'string' && value.length > 20;
                    return (
                      <td key={col} className="px-4 py-2 border-b border-[#E0E5EF] text-sm max-w-xs truncate text-[#233E7D]">
                        {isLong ? (
                          <>
                            {value.slice(0, 20)}...
                            <button
                              className="ml-2 text-[#233E7D] underline text-xs hover:text-[#D82026] font-medium"
                              onClick={() => {
                                setModalContent(value);
                                setModalTitle(col);
                                setModalOpen(true);
                              }}
                            >
                              View More
                            </button>
                          </>
                        ) : (
                          value
                        )}
                      </td>
                    );
                  })}
                </tr>
                {isExpanded && row.Other && (
                  <tr>
                    <td colSpan={columns.length + 1} className="px-4 py-4 border-b border-[#E0E5EF] bg-[#F9FEFF]">
                      <div className="bg-[#fff] rounded-md p-4 shadow-md border border-[#E0E5EF]">
                        <h4 className="text-sm font-semibold text-[#233E7D] mb-3">Additional Details</h4>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                          {Object.entries(row.Other).map(([key, value]) => (
                            <div key={key} className="bg-[#F9FEFF] p-2 rounded border border-[#E0E5EF]">
                              <div className="text-xs font-medium text-[#233E7D] mb-1">{key}</div>
                              <div className="text-sm text-[#233E7D]">{String(value)}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center mt-4 gap-2">
          <button
            className="px-4 py-2 h-10 rounded-md border border-[#233E7D] bg-transparent text-[#233E7D] font-medium text-sm hover:bg-[#233E7D]/10 transition-colors disabled:opacity-50"
            onClick={() => setCurrentPage((p) => Math.max(p - 1, 1))}
            disabled={currentPage === 1}
          >
            Previous
          </button>
          <span className="mx-2 text-sm text-[#6B7280]">
            Page {currentPage} of {totalPages}
          </span>
          <button
            className="px-4 py-2 h-10 rounded-md border border-[#233E7D] bg-transparent text-[#233E7D] font-medium text-sm hover:bg-[#233E7D]/10 transition-colors disabled:opacity-50"
            onClick={() => setCurrentPage((p) => Math.min(p + 1, totalPages))}
            disabled={currentPage === totalPages}
          >
            Next
          </button>
        </div>
      )}

      {/* Modal for viewing full content */}
      {modalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-[#fff] rounded-md shadow-lg max-w-lg w-full p-6 relative border border-[#E0E5EF]">
            <button
              className="absolute top-2 right-2 text-[#6B7280] hover:text-[#D82026] text-xl"
              onClick={() => setModalOpen(false)}
              aria-label="Close"
            >
              &times;
            </button>
            <h3 className="text-lg font-semibold mb-2 text-[#233E7D]">{modalTitle}</h3>
            <div className="text-sm break-words whitespace-pre-wrap max-h-96 overflow-auto text-[#233E7D]">
              {modalContent}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default function UtilisationPage() {
  const [rawMetricsData, setRawMetricsData] = useState<any[]>([]);
  const [rawMetricsLoading, setRawMetricsLoading] = useState(true);
  const [rawMetricsError, setRawMetricsError] = useState<string | null>(null);

  const params = useParams();
  const projectIdRaw = Array.isArray(params?.projectName)
    ? params.projectName[0]
    : params?.projectName;
  const projectId = projectIdRaw ? parseInt(projectIdRaw, 10) : undefined;

  // Fetch raw metrics data once for all charts
  const hasFetched = useRef(false);
  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;
    const fetchRawMetrics = async () => {
      if (!projectId || isNaN(projectId)) return;
      setRawMetricsLoading(true);
      setRawMetricsError(null);
      try {
        const response = await apiQueue.enqueue(
          () => axiosInstance.post(`${BACKEND}/queries_metrics/fetch_raw_metrics`, {
            provider: "azure",
            project_id: projectId,
          }),
          `FetchRawMetrics-${projectId}`
        );
        setRawMetricsData(response.data);
      } catch (err: any) {
        setRawMetricsError("Failed to fetch raw metrics data");
      } finally {
        setRawMetricsLoading(false);
      }
    };
    fetchRawMetrics();
  }, [projectId]);

  if (!projectId || isNaN(projectId)) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="text-red-500">Invalid project ID</div>
      </div>
    );
  }

  if (rawMetricsLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
        <span className="ml-2">Loading utilization data...</span>
      </div>
    );
  }

  if (rawMetricsError) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="text-red-500">{rawMetricsError}</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 bg-[#F9FEFF] min-h-screen py-6 px-2 md:px-8">
      {/* Header */}
      <div className="bg-[#fff] p-6 rounded-md shadow-md border border-[#E0E5EF]">
        <h1 className="text-3xl font-bold text-[#233E7D] mb-2">Azure Utilization Dashboard</h1>
        <p className="text-base text-[#6B7280]">Comprehensive utilization metrics and analytics for your Azure Resources</p>
      </div>

      {/* Azure VM Utilisation Recommendation - Moved to Top */}
      <UtilisationTable data={rawMetricsData} />

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* CPU Utilization Line Chart */}
        <UtilizationLineChart
          data={rawMetricsData}
          title="CPU Utilization Trend"
          metricName="Percentage CPU"
        />

        {/* Instance Type Distribution Pie Chart */}
        <UtilizationPieChart
          data={rawMetricsData}
          title="CPU Usage by Instance Type"
          metricName="Percentage CPU"
          groupBy="instance_type"
        />

        {/* Resource Group Utilization Bar Chart */}
        <UtilizationBarChart
          data={rawMetricsData}
          title="CPU Utilization by Resource Group"
          metricName="Percentage CPU"
          groupBy="resource_group"
          topN={8}
        />

        {/* Region Distribution Pie Chart */}
        <UtilizationPieChart
          data={rawMetricsData}
          title="CPU Usage by Region"
          metricName="Percentage CPU"
          groupBy="resourceregion"
        />
      </div>

      {/* Full Width Components */}
      <div className="space-y-6">
        {/* VM Metrics Table */}
        <UtilizationMetricsTable
          data={rawMetricsData}
          title="Virtual Machine Metrics Summary"
        />
      </div>
    </div>
  );
}
