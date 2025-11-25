"use client";
import React, { useEffect, useState, useRef } from "react";
import { ProjectfetchData } from "@/lib/api";
import apiQueue from "@/lib/apiQueueManager";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ArrowUpDown, Loader2 } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Input } from "@/components/ui/input";

type TableComponentProps = {
  cloudPlatform: string;
  queryType: string;
  projectId: string;
  title: string;
  onDataLoaded?: (data: any[]) => void;
};

interface SortConfig {
  key: string;
  direction: "asc" | "desc";
}

const TableComponent: React.FC<TableComponentProps> = ({
  cloudPlatform,
  queryType,
  projectId,
  title,
  onDataLoaded,
}) => {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [sortConfig, setSortConfig] = useState<SortConfig | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const pathname = usePathname();
  const hasFetched = useRef(false);

  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;
    const getData = async () => {
      let retries = 0;
      const maxRetries = 3;
      while (retries < maxRetries) {
        try {
          if (cloudPlatform && queryType && projectId) {
            const result = await apiQueue.enqueue(
              () => ProjectfetchData(
                cloudPlatform,
                queryType,
                projectId.toString()
              ),
              "ProjectfetchData"
            );

            if (Array.isArray(result)) {
              setData(result);
              if (onDataLoaded) {
                onDataLoaded(result);
              }
            } else {
              console.error("Unexpected data structure:", result);
              setError(new Error("Unexpected data structure"));
            }
            setLoading(false);
            break;
          } else {
            throw new Error("Missing required parameters");
          }
        } catch (error) {
          retries++;
          if (retries === maxRetries) {
            setError(
              error instanceof Error
                ? error
                : new Error("An unknown error occurred")
            );
            setLoading(false);
          }
        }
      }
    };
    getData();
  }, [cloudPlatform, queryType, projectId]);

  const handleSort = (key: string) => {
    setSortConfig((currentConfig) => {
      if (!currentConfig || currentConfig.key !== key) {
        return { key, direction: "asc" };
      }
      if (currentConfig.direction === "asc") {
        return { key, direction: "desc" };
      }
      return null;
    });
  };

  const sortedData = React.useMemo(() => {
    if (!sortConfig) return data;

    return [...data].sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];

      if (aValue === null) return 1;
      if (bValue === null) return -1;

      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortConfig.direction === 'asc' ? aValue - bValue : bValue - aValue;
      }

      const comparison = String(aValue).localeCompare(String(bValue));
      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
  }, [data, sortConfig]);

  const filteredAndSortedData = React.useMemo(() => {
    let filtered = sortedData;
    
    if (searchTerm) {
      filtered = sortedData.filter(item => 
        item.service_name?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    return filtered;
  }, [sortedData, searchTerm]);

  const calculatePercentage = (cost: number) => {
    const totalCost = data.reduce((sum, item) => sum + (item.total_billed_cost || 0), 0);
    return totalCost > 0 ? (cost / totalCost) * 100 : 0;
  };

  if (loading) {
    return (
      <div className="w-full h-48 flex items-center justify-center bg-[#fff] rounded-md shadow-md border border-[#E0E5EF]">
        <Loader2 className="h-8 w-8 animate-spin text-[#233E7D]" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full p-4 text-[#D82026] bg-[#fff] rounded-md shadow-md border border-[#E0E5EF]">
        Error: {error.message}
      </div>
    );
  }

  if (!Array.isArray(data) || data.length === 0) {
    return (
      <div className="w-full p-4 text-[#6B7280] bg-[#fff] rounded-md shadow-md border border-[#E0E5EF]">
        No data available.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex-1 max-w-sm">
          <Input
            placeholder="Search services..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="max-w-sm border-[#E0E5EF] focus:ring-2 focus:ring-[#233E7D]"
          />
        </div>
      </div>
      <div className="rounded-md border border-[#E0E5EF] bg-[#fff] shadow-md">
        <Table>
          <TableHeader className="bg-[#F9FEFF]">
            <TableRow>
              <TableHead 
                className="cursor-pointer w-[35%] text-left text-[#233E7D] font-semibold"
                onClick={() => handleSort("service_name")}
              >
                Service Name{" "}
                {sortConfig?.key === "service_name" && (
                  <ArrowUpDown className="ml-2 h-4 w-4 inline text-[#233E7D]" />
                )}
              </TableHead>
              <TableHead 
                className="cursor-pointer w-[30%] mx-auto text-center text-[#233E7D] font-semibold"
                style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', minWidth: '150px' }}
                onClick={() => handleSort("total_billed_cost")}
              >
                Total Cost{" "}
                {sortConfig?.key === "total_billed_cost" && (
                  <ArrowUpDown className="ml-2 h-4 w-4 inline text-[#233E7D]" />
                )}
              </TableHead>
              {/* <TableHead 
                className="cursor-pointer text-right"
                onClick={() => handleSort("percentage")}
              >
                Percentage{" "}
                {sortConfig?.key === "percentage" && (
                  <ArrowUpDown className="ml-2 h-4 w-4 inline" />
                )}
              </TableHead> */}
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredAndSortedData.map((item, index) => {
              const percentage = calculatePercentage(item.total_billed_cost || 0);
              const serviceName = item.service_name || "Unknown Service";
              const encodedServiceName = encodeURIComponent(serviceName);
              
              return (
                <TableRow key={index} className="hover:bg-[#F9FEFF]">
                  <TableCell className="font-medium text-[#233E7D]">
                    <Link 
                      href={`/connections/${projectId}/${cloudPlatform}/dashboards/${cloudPlatform}dashboard/services/${encodedServiceName}`}
                      className="text-[#233E7D] hover:text-[#D82026] transition-colors duration-200"
                    >
                      {serviceName}
                    </Link>
                  </TableCell>
                  <TableCell className="text-center mx-auto text-[#233E7D] font-semibold" style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', minWidth: '150px' }}>
                    ${item.total_billed_cost != null 
                      ? item.total_billed_cost.toFixed(2).toLocaleString() 
                      : "N/A"}
                  </TableCell>
                  {/* <TableCell className="text-right">
                    {percentage.toFixed(2)}%
                  </TableCell> */}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
};

export default TableComponent;