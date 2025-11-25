"use client";
import React, { useEffect, useState } from "react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { CirclePlus, Trash2, AlertCircle, Pencil } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useParams } from "next/navigation";
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
import axiosInstance, { deleteAlertRule, fetchAlertRules, BACKEND } from "@/lib/api";
import { FaChevronDown, FaChevronUp } from "react-icons/fa";


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
 created_at: string;
 state: Record<string, any>;
 status: boolean;
 resource_list?: string[];
 tag_ids: number[];
 project_ids: number[];
 operation: string;
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
 data: AlertRuleData[];
 onDelete: (alertId: number) => void;
}

const AlertRulesTable: React.FC<AlertRulesTableProps> = ({ data, onDelete }) => {
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


const formatMonitoring = (rule: AlertRuleData) => (
  <div className="space-y-1.5 py-1">
    {/* <div className="grid grid-cols-[80px,1fr] gap-2 items-center">
      <span className="text-gray-500 text-sm">Table:</span>
      <span className="text-sm font-medium truncate" title={rule.table}>
        {rule.table}
      </span>
    </div> */}
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

const [visibleHistoryCount, setVisibleHistoryCount] = useState(3); // State to track how many items are visible

const loadMoreHistory = () => {
  setVisibleHistoryCount(prevCount => prevCount + 3); // Show 3 more items when "Show More" is clicked
};

return (
  <div className="rounded-md border">
    <Table>
      <TableHeader>
        <TableRow className="bg-[#233E7D] hover:bg-[#233E7D]">
          <TableHead className="font-semibold text-white hover:text-white">Alert Name</TableHead>
          <TableHead className="font-semibold text-white hover:text-white">Condition</TableHead>
          <TableHead className="font-semibold text-white hover:text-white">Monitoring</TableHead>
          <TableHead className="font-semibold text-white hover:text-white">Status</TableHead>
          <TableHead className="font-semibold text-white hover:text-white">State</TableHead>
          <TableHead className="font-semibold text-white hover:text-white">Recipient</TableHead>
          <TableHead className="font-semibold text-white hover:text-white">Created</TableHead>
          <TableHead className="font-semibold text-white hover:text-white">State History</TableHead>
          <TableHead className="font-semibold text-white w-[50px] hover:text-white">Action</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {data.map((row) => (
          <TableRow key={row.id} className="hover:bg-[#F9FEFF]">
            <TableCell>
              <div className="font-medium text-[#233E7D]">{row.name}</div>
            </TableCell>
            <TableCell className="min-w-[320px]">
              {formatCondition(row)}
            </TableCell>
            <TableCell className="min-w-[250px]">
              {formatMonitoring(row)}
            </TableCell>
            <TableCell>
              <Badge
                variant="outline"
                className={`px-2 py-1 ${
                  row.status
                    ? "bg-green-50 text-green-700 border-green-200"
                    : "bg-red-50 text-[#D82026] border-[#D82026]/20"
                }`}
              >
                {row.status ? "Active" : "Inactive"}
              </Badge>
            </TableCell>
            <TableCell>
             <Badge
               variant="outline"
               className={`px-2 py-1 ${
                 row.state && Object.keys(row.state)[0]
                   ? row.state[Object.keys(row.state)[0]] === "pending"
                     ? "bg-yellow-50 text-yellow-700 border-yellow-200"
                     : row.state[Object.keys(row.state)[0]] === "firing"
                     ? "bg-red-50 text-[#D82026] border-[#D82026]/20"
                     : row.state[Object.keys(row.state)[0]] === "disabled"
                     ? "bg-gray-50 text-gray-700 border-gray-200"
                     : row.state[Object.keys(row.state)[0]] === "resolved"
                     ? "bg-green-50 text-green-700 border-green-200"
                     : "bg-gray-50 text-gray-700 border-gray-200" // default color for unknown state values
                   : "bg-gray-50 text-gray-700 border-gray-200"
               }`}
             >
               {row.state && Object.keys(row.state)[0]
                 ? row.state[Object.keys(row.state)[0]]
                 : "No state"}
             </Badge>
           </TableCell>
            <TableCell>
              <div className="text-sm font-medium">{row.recipient}</div>
            </TableCell>
            <TableCell>
              <div className="text-sm text-gray-600">
                {new Date(row.created_at).toLocaleDateString()}
                <div className="text-xs text-gray-500">
                  {new Date(row.created_at).toLocaleTimeString()}
                </div>
              </div>
            </TableCell>
            <TableCell>
              <button
                onClick={() => toggleHistoryVisibility(row.id)} // Toggle visibility of history
                className="text-xs text-gray-600 hover:text-gray-700 focus:outline-none underline"
              >
                {historyVisible[row.id] ? "Hide History" : "See History"}
              </button>
              {historyVisible[row.id] && (
              <div className="space-y-1 mt-2">
                {row.state && row.state.history && row.state.history.length > 0 ? (
                  row.state.history
                    .sort((a: HistoryItem, b: HistoryItem) => 
                      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime() // Sorting by timestamp
                    )
                    .slice(0, visibleHistoryCount) // Display only the first `visibleHistoryCount` items
                    .map((historyItem: HistoryItem, index: number) => {
                      return (
                        <div key={index} className="border p-1 rounded-md bg-gray-50">
                          <div className="grid grid-cols-[100px,1fr] gap-1 items-center">
                            <span className="text-gray-500 text-xs">Previous State:</span>
                            <span className="text-xs">{historyItem.previous_state || "N/A"}</span>
                          </div>
                          <div className="grid grid-cols-[100px,1fr] gap-1 items-center">
                            <span className="text-gray-500 text-xs">New State:</span>
                            <span className="text-xs">{historyItem.new_state}</span>
                          </div>
                          <div className="grid grid-cols-[100px,1fr] gap-1 items-center">
                            <span className="text-gray-500 text-xs">Timestamp:</span>
                            <span className="text-xs">{new Date(historyItem.timestamp).toLocaleString()}</span>
                          </div>
                          {historyItem.initial_state !== undefined && (
                            <div className="grid grid-cols-[100px,1fr] gap-1 items-center">
                              <span className="text-gray-500 text-xs">Initial State:</span>
                              <span className="text-xs">{historyItem.initial_state ? "Yes" : "No"}</span>
                            </div>
                          )}
                        </div>
                      );
                    })
                ) : (
                  <div className="text-xs text-gray-500">No history available</div>
                )}

                {row.state.history.length > visibleHistoryCount && (
                  <button
                    onClick={loadMoreHistory}
                    className="mt-2 text-xs text-gray-500 hover:underline"
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
                         className="h-8 w-8 p-0 hover:bg-[#233E7D]/10 hover:text-[#233E7D]"
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
                         className="h-8 w-8 p-0 hover:bg-[#D82026]/10 hover:text-[#D82026]"
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
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete Alert Rule</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete this alert rule? This action
            cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setDeleteConfirmOpen(false)}
            className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10"
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleConfirmDelete}
            className="gap-2 bg-[#D82026] text-white hover:bg-[#b81a1f]"
          >
            <Trash2 className="h-4 w-4" />
            Delete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    
    <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Edit Alert Rule</DialogTitle>
            <DialogDescription>
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
const params = useParams();
const projectId = params.projectName as string;




const filterSections: FilterSectionProps[] = [];




useEffect(() => {
  const getAlertRules = async () => {
    try {
      const data = await fetchAlertRules(projectId);
      const sortedData = [...data].sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
      setAlertRules(sortedData);
      setFilteredAlertRules(sortedData);
    } catch (error) {
      console.error("Error fetching alert rules:", error);
    }
  };




  getAlertRules();
}, [projectId, refreshTrigger]);




useEffect(() => {
  const filtered = alertRules.filter((rule) =>
    rule.name.toLowerCase().includes(searchQuery.toLowerCase())
  );
  setFilteredAlertRules(filtered);
}, [searchQuery, alertRules]);




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




return (
  <div className="bg-white rounded-lg shadow-sm border p-6 w-full">
    <h1 className="text-2xl font-semibold text-[#233E7D] mb-6">Alert Rules</h1>



    {!showNewAlertForm ? (
      <div className="space-y-6">
        {filterSections.length > 0 && (
          <div className="flex gap-6">
            {filterSections.map((section) => (
              <FilterSection key={section.title} {...section} />
            ))}
          </div>
        )}




        <div className="flex justify-between items-center gap-4">
          <div className="flex-1 max-w-md">
            <Input
              placeholder="Search alert rules..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full"
            />
          </div>
          <Button
            variant="default"
            className="gap-2 bg-[#D82026] text-white hover:bg-[#b81a1f]"
            onClick={handleNewAlertClick}
          >
            <CirclePlus className="h-4 w-4" />
            New Alert Rule
          </Button>
        </div>
        <AlertRulesTable
          data={filteredAlertRules}
          onDelete={handleDeleteAlert}
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
);
};

export default AlertRulesPage;