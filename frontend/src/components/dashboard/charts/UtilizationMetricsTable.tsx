import React, { useMemo, useState } from "react";
import { Table, ChevronDown, ChevronRight } from "lucide-react";

interface UtilizationDataPoint {
  vm_name: string;
  resource_group: string;
  subscription_id: string;
  timestamp: string;
  value: number;
  metric_name: string;
  unit: string;
  displaydescription: string;
  namespace: string;
  resourceregion: string;
  resource_id: string;
  instance_type: string;
}

interface VMSummary {
  vm_name: string;
  resource_group: string;
  resourceregion: string;
  instance_type: string;
  metrics: {
    [metricName: string]: {
      average: number;
      minimum: number;
      maximum: number;
      unit: string;
      count: number;
    };
  };
}

interface UtilizationMetricsTableProps {
  data: UtilizationDataPoint[];
  title: string;
}

const UtilizationMetricsTable: React.FC<UtilizationMetricsTableProps> = ({
  data,
  title,
}) => {
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [sortColumn, setSortColumn] = useState<string>('vm_name');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const tableData = useMemo(() => {
    if (data.length === 0) return [];

    // Group data by VM
    const groupedData = data.reduce((acc, item) => {
      const vmKey = `${item.vm_name}-${item.resource_group}`;
      if (!acc[vmKey]) {
        acc[vmKey] = {
          vm_name: item.vm_name,
          resource_group: item.resource_group,
          resourceregion: item.resourceregion,
          instance_type: item.instance_type,
          metrics: {},
        };
      }

      // Group metrics
      const metricName = item.metric_name;
      if (!acc[vmKey].metrics[metricName]) {
        acc[vmKey].metrics[metricName] = {
          values: [],
          unit: item.unit,
        };
      }
      acc[vmKey].metrics[metricName].values.push(item.value);
      return acc;
    }, {} as any);

    // Calculate statistics for each metric
    const processedData: VMSummary[] = Object.values(groupedData).map((vm: any) => {
      const processedMetrics: { [key: string]: any } = {};
      
      Object.entries(vm.metrics).forEach(([metricName, metricData]: [string, any]) => {
        const values = metricData.values.sort((a: number, b: number) => a - b);
        const sum = values.reduce((total: number, val: number) => total + val, 0);
        const unitText = metricData.unit || "";
        const formattedUnit = unitText === 'Percent' ? '%' : unitText;
        
        processedMetrics[metricName] = {
          average: parseFloat((sum / values.length).toFixed(2)),
          minimum: parseFloat(values[0].toFixed(2)),
          maximum: parseFloat(values[values.length - 1].toFixed(2)),
          unit: formattedUnit,
          count: values.length,
        };
      });

      return {
        vm_name: vm.vm_name,
        resource_group: vm.resource_group,
        resourceregion: vm.resourceregion,
        instance_type: vm.instance_type,
        metrics: processedMetrics,
      };
    });

    // Sort data
    processedData.sort((a, b) => {
      let aVal: any = a[sortColumn as keyof VMSummary];
      let bVal: any = b[sortColumn as keyof VMSummary];
      
      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase();
        bVal = bVal.toLowerCase();
      }
      
      if (sortDirection === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      }
    });

    return processedData;
  }, [data, sortColumn, sortDirection]);

  const toggleRowExpansion = (vmKey: string) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(vmKey)) {
      newExpanded.delete(vmKey);
    } else {
      newExpanded.add(vmKey);
    }
    setExpandedRows(newExpanded);
  };

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  if (tableData.length === 0) {
    return (
      <div className="bg-[#fff] p-6 rounded-md shadow-md border border-[#E0E5EF] h-96 flex flex-col items-center justify-center">
        <Table className="w-12 h-12 text-[#C8C8C8] mb-4" />
        <h3 className="text-lg font-semibold text-[#233E7D] mb-2">{title}</h3>
        <p className="text-[#6B7280] text-center">
          No utilization data available
        </p>
      </div>
    );
  }

  // Get unique metrics for header
  const allMetrics = Array.from(
    new Set(tableData.flatMap(vm => Object.keys(vm.metrics)))
  );

  return (
    <div className="bg-[#fff] p-6 rounded-md shadow-md border border-[#E0E5EF]">
      <h3 className="text-lg font-semibold text-[#233E7D] mb-4">{title}</h3>
      <div className="overflow-x-auto">
        <table className="min-w-full table-auto border border-[#E0E5EF] rounded-md">
          <thead className="bg-[#F9FEFF]">
            <tr>
              <th className="px-4 py-2 text-left text-xs font-semibold text-[#233E7D] uppercase tracking-wider border-b border-[#E0E5EF]">
                Expand
              </th>
              <th 
                className="px-4 py-2 text-left text-xs font-semibold text-[#233E7D] uppercase tracking-wider border-b border-[#E0E5EF] cursor-pointer hover:bg-[#233E7D]/5"
                onClick={() => handleSort('vm_name')}
              >
                VM Name {sortColumn === 'vm_name' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                className="px-4 py-2 text-left text-xs font-semibold text-[#233E7D] uppercase tracking-wider border-b border-[#E0E5EF] cursor-pointer hover:bg-[#233E7D]/5"
                onClick={() => handleSort('instance_type')}
              >
                Instance Type {sortColumn === 'instance_type' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                className="px-4 py-2 text-left text-xs font-semibold text-[#233E7D] uppercase tracking-wider border-b border-[#E0E5EF] cursor-pointer hover:bg-[#233E7D]/5"
                onClick={() => handleSort('resourceregion')}
              >
                Region {sortColumn === 'resourceregion' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th className="px-4 py-2 text-left text-xs font-semibold text-[#233E7D] uppercase tracking-wider border-b border-[#E0E5EF]">
                CPU Avg
              </th>
            </tr>
          </thead>
          <tbody className="bg-[#fff] divide-y divide-[#E0E5EF]">
            {tableData.map((vm) => {
              const vmKey = `${vm.vm_name}-${vm.resource_group}`;
              const isExpanded = expandedRows.has(vmKey);
              const cpuMetric = vm.metrics["Percentage CPU"];
              
              return (
                <React.Fragment key={vmKey}>
                  <tr className="hover:bg-[#233E7D]/5 transition-colors">
                    <td className="px-4 py-2 whitespace-nowrap">
                      <button
                        onClick={() => toggleRowExpansion(vmKey)}
                        className={`transition-colors rounded hover:bg-[#E0E5EF] p-1 ${isExpanded ? 'text-[#D82026]' : 'text-[#233E7D]'}`}
                        aria-label={isExpanded ? 'Collapse details' : 'Expand details'}
                      >
                        {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                      </button>
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-[#233E7D]">
                      <span 
                        className="truncate max-w-32 block" 
                        title={vm.vm_name}
                      >
                        {vm.vm_name}
                      </span>
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-[#6B7280]">
                      <span 
                        className="truncate max-w-24 block" 
                        title={vm.instance_type}
                      >
                        {vm.instance_type}
                      </span>
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-[#6B7280]">
                      <span 
                        className="truncate max-w-20 block" 
                        title={vm.resourceregion}
                      >
                        {vm.resourceregion}
                      </span>
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-[#6B7280]">
                      {cpuMetric ? `${cpuMetric.average} ${cpuMetric.unit}` : 'N/A'}
                    </td>
                  </tr>
                  {isExpanded && (
                    <tr>
                      <td colSpan={5} className="px-4 py-2 bg-[#F9FEFF] border-b border-[#E0E5EF]">
                        <div className="space-y-2">
                          <div className="text-sm text-[#233E7D]">
                            <strong>Resource Group:</strong> 
                            <span className="ml-1" title={vm.resource_group}>
                              {vm.resource_group.length > 50 ? `${vm.resource_group.slice(0, 50)}...` : vm.resource_group}
                            </span>
                          </div>
                          <div className="grid grid-cols-2 gap-4">
                            {allMetrics.map(metricName => {
                              const metric = vm.metrics[metricName];
                              if (!metric) return null;
                              
                              return (
                                <div key={metricName} className="text-sm bg-[#fff] border border-[#E0E5EF] rounded-md p-3">
                                  <div 
                                    className="font-semibold text-[#233E7D] truncate" 
                                    title={metricName}
                                  >
                                    {metricName.length > 25 ? `${metricName.slice(0, 25)}...` : metricName}
                                  </div>
                                  <div className="text-[#6B7280] space-y-1">
                                    <div>Avg: <span className="font-semibold text-[#233E7D]">{metric.average} {metric.unit}</span></div>
                                    <div>Min: <span className="font-semibold text-[#233E7D]">{metric.minimum} {metric.unit}</span></div>
                                    <div>Max: <span className="font-semibold text-[#233E7D]">{metric.maximum} {metric.unit}</span></div>
                                    <div>Samples: <span className="font-semibold text-[#233E7D]">{metric.count}</span></div>
                                  </div>
                                </div>
                              );
                            })}
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
      </div>
      <div className="mt-4 text-sm text-[#6B7280]">
        Showing <span className="font-semibold text-[#233E7D]">{tableData.length}</span> VMs with utilization data
      </div>
    </div>
  );
};

export default UtilizationMetricsTable;
