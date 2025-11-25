"use client";
import Link from "next/link";
import React, { useEffect, useState } from "react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { CirclePlus, Trash2, AlertCircle, X, Pencil } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useParams } from "next/navigation";
import { newFetchProjectDetails } from "@/lib/api";
import { FaChevronDown, FaChevronUp } from "react-icons/fa";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import NewAlertRuleForm from "@/components/alertRules/NewAlertRuleForm";
import axiosInstance, { deleteAlertRule, fetchAllAlertRules, BACKEND } from "@/lib/api";

interface AlertRuleData {
 id: number;
 name: string;
 alert_type: string;
 condition: string;
 type: string;
 value_threshold?: string;
 percentage_threshold?: string;
 recipient: string;
 ends_on: string;
 schedule: string;
 operation:string;
 created_at: string;
 state: Record<string, any>;
 status: boolean;
 resource_list?: string[];
 tag_ids: number[];
 project_ids: number[];
}

interface AlertRulesTableProps {
  data: AlertRuleData[];
  onDelete: (alertId: number) => void;
}

interface HistoryItem {
  previous_state: string | null;
  new_state: string;
  timestamp: string; // Assuming timestamp is in ISO string format
  initial_state?: boolean; // This is optional, because it's not present in every entry
}

// Define the shape of form data that NewAlertRuleForm will return
interface AlertRuleFormData {
  name: string;
  alert_type: string;
  condition: string;
  type: string;
  value_threshold?: string;
  percentage_threshold?: string;
  recipient: string;
  ends_on: string;
  schedule: string;
  resource_list?: string[];
  tag_ids: number[];
  project_ids: number[];
  operation: string;
 } 

interface FilterSectionProps {
 title: string;
 options: string[];
}

const FilterSection: React.FC<FilterSectionProps> = ({ title, options }) => (
  <div>
    <span className="text-sm font-medium text-gray-700">{title}</span>
    <div className="bg-slate-50 p-1.5 rounded-md mt-1">
      <ToggleGroup type="single" className="gap-1">
        {options.map((option) => (
          <ToggleGroupItem
            key={option}
            value={option}
            className="data-[state=on]:bg-white data-[state=on]:shadow-sm"
          >
            {option}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  </div>
  );


const updateAlertRule = async (alertId: number, data: AlertRuleFormData): Promise<AlertRuleData> => {
  try {
    const response = await axiosInstance.put(`${BACKEND}/alerts/${alertId}`, data, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    return response.data;
  } catch (error) {
    throw new Error('Failed to update alert rule');
  }
 };
  
interface AlertRulesTableProps {
  data: AlertRuleData[];  // Array of alert rule data
  onDelete: (alertId: number) => void;  // Function to handle delete action
  projectNames: Record<number, string>;  // Mapping of project IDs to project names
  projectCloudPlatforms: string;  // Single cloud platform for all projects, as a string
}

const AlertRulesTable: React.FC<AlertRulesTableProps> = ({ data, onDelete, projectNames, projectCloudPlatforms }) => {
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [alertToDelete, setAlertToDelete] = useState<number | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingAlert, setEditingAlert] = useState<AlertRuleData | null>(null);
  const [historyVisible, setHistoryVisible] = useState<{ [key: number]: boolean }>({}); // Track history visibility for each alert

  const handleDeleteClick = (alertId: number) => {
    setAlertToDelete(alertId);
    setDeleteConfirmOpen(true);
  };

  const handleEditClick = (alert: AlertRuleData) => {
    setEditingAlert(alert);
    setEditDialogOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (alertToDelete !== null) {
      try {
        await deleteAlertRule(alertToDelete);
        onDelete(alertToDelete);
        setDeleteConfirmOpen(false);
      } catch (error) {
        console.error("Failed to delete alert rule:", error);
      }
    }
  };

  const handleUpdateAlert = () => {
    if (!editingAlert) return;
    
    updateAlertRule(editingAlert.id, editingAlert)
      .then(() => {
        window.location.reload();
        setEditDialogOpen(false);
      })
      .catch((error) => {
        console.error("Failed to update alert rule:", error);
      });
  };

  const toggleHistoryVisibility = (alertId: number) => {
    setHistoryVisible(prevState => ({
      ...prevState,
      [alertId]: !prevState[alertId],
    }));
  };

  const AlertRuleConditions = ({ rule }: { rule: AlertRuleData }) => {
    const [firstDropdownOpen, setFirstDropdownOpen] = useState(false);
    const [secondDropdownOpen, setSecondDropdownOpen] = useState(false);
    const [visibleResources, setVisibleResources] = useState(3);

    const handleShowMoreResources = () => {
      setVisibleResources(prev => prev + 3);
    };

    const hasMoreResources = rule.resource_list && rule.resource_list.length > visibleResources;
  
    return (
      <div className="space-y-1.5 py-1">
        {/* First Dropdown */}
        <div>
          <button
            className="w-full text-left text-black-600 flex items-center justify-between"
            onClick={() => setFirstDropdownOpen(!firstDropdownOpen)}
          >
            <span>Alert Configuration</span>
            <span>
              {firstDropdownOpen ? (
                <FaChevronUp className="text-gray-500" />
              ) : (
                <FaChevronDown className="text-gray-500" />
              )}
            </span>
          </button>
          {firstDropdownOpen && (
            <div className="space-y-1.5 py-1 mt-2">
              {rule.alert_type && (
                <div className="grid grid-cols-[100px,1fr] gap-2 items-center">
                  <span className="text-gray-500 text-sm">Alert Type:</span>
                  <span className="text-sm font-medium">{rule.alert_type}</span>
                </div>
              )}
              {rule.operation && (
                <div className="grid grid-cols-[100px,1fr] gap-2 items-center">
                  <span className="text-gray-500 text-sm">Operation:</span>
                  <span className="text-sm font-medium">{rule.operation}</span>
                </div>
              )}
              {rule.condition && (
                <div className="grid grid-cols-[100px,1fr] gap-2 items-center">
                  <span className="text-gray-500 text-sm">Condition:</span>
                  <span className="text-sm font-medium">{rule.condition}</span>
                </div>
              )}
              {rule.value_threshold && (
                <div className="grid grid-cols-[100px,1fr] gap-2 items-center">
                  <span className="text-gray-500 text-sm">Threshold:</span>
                  <span className="text-sm font-medium">{rule.value_threshold}</span>
                </div>
              )}
              {rule.percentage_threshold && (
                <div className="grid grid-cols-[100px,1fr] gap-2 items-center">
                  <span className="text-gray-500 text-sm">Threshold:</span>
                  <span className="text-sm font-medium">{rule.percentage_threshold}</span>
                </div>
              )}
            </div>
          )}
        </div>
  
        {/* Second Dropdown */}
        <div>
          <button
            className="w-full text-left text-black-600 flex items-center justify-between"
            onClick={() => setSecondDropdownOpen(!secondDropdownOpen)}
          >
            <span>Projects, Resources & Tags</span>
            <span>
              {secondDropdownOpen ? (
                <FaChevronUp className="text-gray-500" />
              ) : (
                <FaChevronDown className="text-gray-500" />
              )}
            </span>
          </button>
          {secondDropdownOpen && (
            <div className="space-y-1.5 py-1 mt-2">
              {rule.resource_list && rule.resource_list.length > 0 && (
                <div className="space-y-2">
                  <div className="grid grid-cols-[120px,1fr] gap-2">
                    <span className="text-gray-500 text-sm">Resources:</span>
                    <div className="space-y-1">
                      {rule.resource_list.slice(0, visibleResources).map((resource, idx) => (
                        <span key={idx} className="mr-1 text-xs text-gray-700">
                          {resource} {idx < visibleResources - 1 && <span>, </span>}
                        </span>
                      ))}
                      {hasMoreResources && (
                        <span
                          onClick={handleShowMoreResources}
                          className="mt-2 text-xs text-gray-500 hover:text-gray-700 cursor-pointer underline"
                        >
                          Show More Resources
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}
              {rule.project_ids && rule.project_ids.length > 0 && (
                <div className="grid grid-cols-[100px,1fr] gap-2 items-center">
                  <span className="text-gray-500 text-sm">Projects:</span>
                  <span className="text-sm font-medium">
                    {rule.project_ids.map((id) => (
                      <div key={id} className="flex items-center space-x-2">
                        <Link
                          href={`/connections/${id}/${projectCloudPlatforms}`} 
                          className="text-blue-600 hover:underline"
                        >
                          {projectNames[id] || `Project ID: ${id}`}
                        </Link>
                        <span className="text-gray-400 text-sm">
                          ({projectCloudPlatforms || "Unknown Platform"})
                        </span>
                      </div>
                    ))}
                  </span>
                </div>
              )}
              {rule.tag_ids && rule.tag_ids.length > 0 && (
                <div className="grid grid-cols-[100px,1fr] gap-2 items-center">
                  <span className="text-gray-500 text-sm">Tags:</span>
                  <span className="text-sm font-medium">{rule.tag_ids}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };
  
  const formatCondition = (rule: AlertRuleData) => {
    return <AlertRuleConditions rule={rule} />;
  };

  const [visibleHistoryCount, setVisibleHistoryCount] = useState(3); // State to track how many items are visible

const loadMoreHistory = () => {
  setVisibleHistoryCount(prevCount => prevCount + 3); // Show 3 more items when "Show More" is clicked
};

 const formatMonitoring = (rule: AlertRuleData) => (
   <div className="space-y-1.5 py-1">
     <div className="grid grid-cols-[80px,1fr] gap-2 items-center">
       <span className="text-gray-500 text-sm">Schedule:</span>
       <span className="text-sm font-medium">{rule.schedule}</span>
     </div>
     <div className="grid grid-cols-[80px,1fr] gap-2 items-center">
       <span className="text-gray-500 text-sm">Ends On:</span>
       <span className="text-sm font-medium">{rule.ends_on}</span>
     </div>
   </div>
 );

 return (
   <div className="rounded-md border border-[#E0E5EF] bg-white shadow-md">
     <Table>
       <TableHeader>
         <TableRow className="bg-[#233E7D] hover:bg-[#233E7D]">
           <TableHead className="font-semibold text-white text-base hover:text-white">Alert Name</TableHead>
           <TableHead className="font-semibold text-white text-base hover:text-white">Details</TableHead>
           <TableHead className="font-semibold text-white text-base hover:text-white">Monitoring</TableHead>
           <TableHead className="font-semibold text-white text-base hover:text-white">Status</TableHead>
           <TableHead className="font-semibold text-white text-base hover:text-white">State</TableHead>
           <TableHead className="font-semibold text-white text-base hover:text-white">Recipient</TableHead>
           <TableHead className="font-semibold text-white text-base hover:text-white">Created</TableHead>
           <TableHead className="font-semibold text-white text-base hover:text-white">State History</TableHead>
           <TableHead className="font-semibold text-white text-base w-[50px] hover:text-white">Action</TableHead>
         </TableRow>
       </TableHeader>
       <TableBody>
         {data.map((row) => (
           <TableRow key={row.id} className="border-b border-[#E0E5EF]">
              {/* Removed hover:bg-[#233E7D]/5 to disable white/hover effect */}
              <TableCell>
                <div className="font-semibold text-[#233E7D] text-base">{row.name}</div>
              </TableCell>
              <TableCell className="min-w-[320px] text-[#233E7D] text-base">
                {formatCondition(row)}
              </TableCell>
              <TableCell className="min-w-[250px] text-[#233E7D] text-base">
                {formatMonitoring(row)}
              </TableCell>
              <TableCell>
                <Badge
                  variant="outline"
                  className={`px-2 py-1 text-sm font-medium rounded-md border ${
                    row.status
                      ? "bg-[#F0FDF4] text-[#22C55E] border-[#BBF7D0]"
                      : "bg-[#FEF2F2] text-[#D82026] border-[#FECACA]"
                  }`}
                >
                  {row.status ? "Active" : "Inactive"}
                </Badge>
              </TableCell>
              <TableCell>
                <Badge
                  variant="outline"
                  className={`px-2 py-1 text-sm font-medium rounded-md border ${
                    row.state && Object.keys(row.state)[0]
                      ? row.state[Object.keys(row.state)[0]] === "pending"
                        ? "bg-yellow-50 text-yellow-700 border-yellow-200"
                        : row.state[Object.keys(row.state)[0]] === "firing"
                        ? "bg-[#FEF2F2] text-[#D82026] border-[#FECACA]"
                        : row.state[Object.keys(row.state)[0]] === "disabled"
                        ? "bg-[#F9FAFB] text-[#6B7280] border-[#E0E5EF]"
                        : row.state[Object.keys(row.state)[0]] === "resolved"
                        ? "bg-[#F0FDF4] text-[#22C55E] border-[#BBF7D0]"
                        : "bg-[#F9FAFB] text-[#6B7280] border-[#E0E5EF]"
                      : "bg-[#F9FAFB] text-[#6B7280] border-[#E0E5EF]"
                  }`}
                >
                  {row.state && Object.keys(row.state)[0]
                    ? row.state[Object.keys(row.state)[0]]
                    : "No state"}
                </Badge>
              </TableCell>
              <TableCell className="text-[#6B7280] text-base">{row.recipient}</TableCell>
              <TableCell className="text-[#6B7280] text-base">{new Date(row.created_at).toLocaleString()}</TableCell>
              <TableCell>
                <button
                  onClick={() => toggleHistoryVisibility(row.id)}
                  className="text-xs text-[#233E7D] hover:text-[#D82026] focus:outline-none underline"
                >
                  {historyVisible[row.id] ? "Hide History" : "See History"}
                </button>
                {historyVisible[row.id] && (
                  <div className="space-y-1 mt-2">
                    {row.state && row.state.history && row.state.history.length > 0 ? (
                      row.state.history
                        .sort((a: HistoryItem, b: HistoryItem) =>
                          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
                        )
                        .slice(0, visibleHistoryCount)
                        .map((historyItem: HistoryItem, index: number) => {
                          return (
                            <div key={index} className="border border-[#E0E5EF] p-1 rounded-md bg-[#F9FEFF]">
                              <div className="grid grid-cols-[100px,1fr] gap-1 items-center">
                                <span className="text-[#6B7280] text-xs">Previous State:</span>
                                <span className="text-xs">{historyItem.previous_state || "N/A"}</span>
                              </div>
                              <div className="grid grid-cols-[100px,1fr] gap-1 items-center">
                                <span className="text-[#6B7280] text-xs">New State:</span>
                                <span className="text-xs">{historyItem.new_state}</span>
                              </div>
                              <div className="grid grid-cols-[100px,1fr] gap-1 items-center">
                                <span className="text-[#6B7280] text-xs">Timestamp:</span>
                                <span className="text-xs">{new Date(historyItem.timestamp).toLocaleString()}</span>
                              </div>
                              {historyItem.initial_state !== undefined && (
                                <div className="grid grid-cols-[100px,1fr] gap-1 items-center">
                                  <span className="text-[#6B7280] text-xs">Initial State:</span>
                                  <span className="text-xs">{historyItem.initial_state ? "Yes" : "No"}</span>
                                </div>
                              )}
                            </div>
                          );
                        })
                    ) : (
                      <div className="text-xs text-[#6B7280]">No history available</div>
                    )}

                    {row.state.history.length > visibleHistoryCount && (
                      <button
                        onClick={loadMoreHistory}
                        className="mt-2 text-xs text-[#233E7D] hover:text-[#D82026] underline"
                      >
                        Show More
                      </button>
                    )}
                  </div>
                )}
              </TableCell>
              <TableCell>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 p-0 hover:bg-[#233E7D]/10 hover:text-[#D82026] text-[#233E7D]"
                        onClick={() => handleEditClick(row)}
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Edit alert rule</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 p-0 hover:bg-[#D82026]/10 hover:text-[#D82026] text-[#233E7D]"
                        onClick={() => handleDeleteClick(row.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Delete alert rule</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </TableCell>
            </TableRow>
         ))}
       </TableBody>
     </Table>
     <Dialog open={deleteConfirmOpen} onOpenChange={setDeleteConfirmOpen}>
       <DialogContent className="rounded-md bg-white">
         <DialogHeader>
           <DialogTitle className="text-[#D82026] text-2xl font-bold">Confirm Deletion</DialogTitle>
           <DialogDescription className="text-base text-[#6B7280]">
             Are you sure you want to delete this alert rule?
           </DialogDescription>
         </DialogHeader>
         <DialogFooter>
           <Button
             variant="outline"
             onClick={() => setDeleteConfirmOpen(false)}
             className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10 text-sm font-medium rounded-md"
           >
             Cancel
           </Button>
           <Button
             className="ml-2 bg-[#D82026] text-white hover:bg-[#b81a1f] text-sm font-medium rounded-md"
             onClick={handleConfirmDelete}
           >
             Delete
           </Button>
         </DialogFooter>
       </DialogContent>
     </Dialog>
     <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent className="max-w-4xl rounded-md bg-white">
          <DialogHeader>
            <DialogTitle className="text-[#233E7D] text-2xl font-bold">Edit Alert Rule</DialogTitle>
            <DialogDescription className="text-base text-[#6B7280]">
              Update the details of your alert rule.
            </DialogDescription>
          </DialogHeader>
          {editingAlert && (
            <NewAlertRuleForm
              onBack={() => setEditDialogOpen(false)}
              onSave={handleUpdateAlert}
              projectId={editingAlert.project_ids[0].toString()}
            />
          )}
        </DialogContent>
      </Dialog>
   </div>
 );
};

const AlertRulesPage: React.FC = () => {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [alertRules, setAlertRules] = useState<AlertRuleData[]>([]);
  const [filteredAlertRules, setFilteredAlertRules] = useState<AlertRuleData[]>([]);
  const [showNewAlertForm, setShowNewAlertForm] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const params = useParams();
  const projectId = params.projectName as string;
  const [projectNames, setProjectNames] = useState<Record<number, string>>({}); // Moved here
  const [projectCloudPlatforms, setProjectCloudPlatforms] = useState<string>("");

  // Dynamically gather the alert types from the fetched data
  const alertTypes = Array.from(new Set(alertRules.map(rule => rule.type)));

  useEffect(() => {
    const getAlertRules = async () => {
      try {
        const data = await fetchAllAlertRules();
        const sortedData = [...data].sort(
          (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        );
        setAlertRules(sortedData);
        setFilteredAlertRules(sortedData);

        // Fetch project names and cloud platforms for all unique project IDs in alert rules
        const uniqueProjectIds = Array.from(new Set(sortedData.flatMap((rule) => rule.project_ids)));
        
        uniqueProjectIds.forEach(async (id) => {
          if (!projectNames[id] || !projectCloudPlatforms[id]) {  // Only fetch if not already fetched
            try {
              const projectDetails = await newFetchProjectDetails(String(id));
              setProjectNames((prev) => ({ ...prev, [id]: projectDetails.name }));
              setProjectCloudPlatforms(projectDetails.cloudPlatform);
            } catch (error) {
              console.error(`Failed to fetch project details for ID ${id}:`, error);
            }
          }
        });
      } catch (error) {
        console.error("Error fetching alert rules:", error);
      }
    };

    getAlertRules();
  }, [refreshTrigger]);

  useEffect(() => {
    const filtered = alertRules.filter((rule) =>
      rule.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
      (selectedType ? rule.type === selectedType : true)
    );
    setFilteredAlertRules(filtered);
  }, [searchQuery, alertRules, selectedType]);

  const handleNewAlertClick = () => setShowNewAlertForm(true);
  const handleBackClick = () => {
    setShowNewAlertForm(false);
    setRefreshTrigger((prev) => prev + 1);
  };

  const handleDeleteAlert = (deletedAlertId: number) => {
    setAlertRules((prevRules) =>
      prevRules.filter((rule) => rule.id !== deletedAlertId)
    );
    setFilteredAlertRules((prevRules) =>
      prevRules.filter((rule) => rule.id !== deletedAlertId)
    );
  };

  const handleTypeSelect = (type: string) => {
    setSelectedType(type === selectedType ? null : type);
  };

  const colors = [
    "bg-[#FEF9C3]",   // yellow-100
    "bg-[#DCFCE7]",   // green-100
    "bg-[#F3E8FF]",   // purple-100
    "bg-[#FECACA]",   // red-100
    "bg-[#DBEAFE]",   // blue-100
  ];

  return (
    <div className="bg-[#F9FEFF] min-h-screen p-8">
      <div className="bg-white rounded-md shadow-md border border-[#E0E5EF] p-6 w-full">
        <h1 className="text-3xl font-bold text-[#233E7D] mb-6">Alert Rules</h1>

        {!showNewAlertForm ? (
          <div className="space-y-6">
            <div className="flex items-center space-x-4">
              <ToggleGroup type="single" value={selectedType || ''} onValueChange={handleTypeSelect}>
                {alertTypes.map((type, index) => (
                  <ToggleGroupItem
                    key={type}
                    value={type}
                    className={`relative px-3 py-2 text-sm font-medium rounded-md m-1
                      ${selectedType === type
                        ? "bg-[#233E7D] text-white"
                        : colors[index % colors.length] + " text-[#233E7D]"
                      }
                      data-[state=on]:bg-[#233E7D] data-[state=on]:text-white
                    `}
                  >
                    {type}
                    {selectedType === type && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="absolute -top-2 -right-2 h-5 w-5 p-0 rounded-full"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedType(null);
                        }}
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    )}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </div>

            <div className="flex justify-between items-center gap-4">
              <div className="flex-1 max-w-md">
                <Input
                  placeholder="Search alert rules..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full border-[#C8C8C8] rounded-md px-3 py-2 placeholder:text-gray-400 focus:ring-2 focus:ring-[#233E7D] text-base"
                />
              </div>
              <Button
                className="gap-2 bg-[#D82026] hover:bg-[#b81a1f] text-white text-sm font-medium rounded-md h-10 px-4 py-2"
                onClick={handleNewAlertClick}
              >
                <CirclePlus className="h-4 w-4" />
                New Alert Rule
              </Button>
            </div>

            <AlertRulesTable
              data={filteredAlertRules}
              onDelete={handleDeleteAlert}
              projectNames={projectNames}
              projectCloudPlatforms={projectCloudPlatforms}
            />
          </div>
        ) : (
          <NewAlertRuleForm
            onBack={handleBackClick}
            onSave={handleBackClick}
            projectId={projectId}
          />
        )}
      </div>
    </div>
  );
};

export default AlertRulesPage;
