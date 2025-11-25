import React, { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Switch } from "@/components/ui/switch";
import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import { ChevronDownIcon } from "@heroicons/react/20/solid";
import { useToast } from "@/components/ui/use-toast";
import { Loader } from "lucide-react";
import axiosInstance, { saveAlertRule, BACKEND } from "@/lib/api";
import { addDays } from "date-fns";
import { format } from "date-fns";
import { Calendar } from "@/components/ui/calendar";
import {
 Popover,
 PopoverContent,
 PopoverTrigger,
} from "@/components/ui/popover";
import { CalendarIcon } from "lucide-react";
import { cn } from "@/lib/utils";


// Define configurations for different alert types
const alertTypeConfigs = {
 Cost: {
   requiredFields: ['operation', 'condition', 'threshold', 'schedule', 'recipient'],
   operations: ['SUM', 'AVERAGE'],
   conditions: [
     { label: "Greater than", value: "Greater than" },
     { label: "Less than", value: "Less than" },
     { label: "Equal to", value: "Equal to" },
     { label: "Not equal to", value: "Not equal to" },
     { label: "Greater than equal to", value: "Greater than equal to" },
     { label: "Less than equal to", value: "Less than equal to" }
   ],
   thresholdType: 'number',
   scheduleOptions: ['Daily', 'Weekly', 'Monthly']
 },
 Spike: {
   requiredFields: ['threshold', 'schedule', 'recipient'],
   operations: ['SUM', 'AVERAGE'],
   conditions: [
     { label: "Greater than", value: "Greater than" },
     { label: "Less than", value: "Less than" },
     { label: "Equal to", value: "Equal to" }
   ],
   thresholdType: 'number',
   scheduleOptions: ['Daily', 'Weekly', 'Monthly']
 },
};


interface Integration {
 id: number;
 name: string;
}


interface AlertRuleData {
 name: string;
 recipient: string;
 integration_id?: number;
 ends_on: string;
 schedule: string;
 status: boolean;
 type: string;
 alert_type: string;
 resource_list: any[];
 percentage_threshold?: number | null;
 value_threshold?: number | null;
 condition: string;
 operation: string;
 state: Record<string, any>;
 tag_ids: number[];
 project_ids: number[];
}


interface NewAlertRuleFormProps {
 onBack: () => void;
 onSave: () => void;
 projectId: string;
 selectedTagIds?: number[];
 selectedResourceNames?: string[];
 initialValues?: AlertRuleData;
}


const NewAlertRuleForm: React.FC<NewAlertRuleFormProps> = ({
 onBack,
 onSave,
 projectId,
 selectedTagIds = [],
 selectedResourceNames = [],
 initialValues
}) => {
 const { toast } = useToast();


 // State management for form inputs
 const [alertName, setAlertName] = useState(initialValues?.name || "");
 const [alertEndsIn, setAlertEndsIn] = useState("1-Week");
 const [alertStatus, setAlertStatus] = useState(initialValues?.status || false);
 const [alertType, setAlertType] = useState(initialValues?.alert_type || "Cost");
 const [integrations, setIntegrations] = useState<Integration[]>([]);
 const [selectedRecipient, setSelectedRecipient] = useState(initialValues?.recipient || "");
 const [selectedIntegration, setSelectedIntegration] = useState<Integration | null>(null);
 const [selectedFrequency, setSelectedFrequency] = useState(initialValues?.schedule || "");
 const [loading, setLoading] = useState(false);
 const [selectedOperation, setSelectedOperation] = useState(initialValues?.operation || "");
 const [selectedType, setSelectedType] = useState(initialValues?.type || "Connection");
 const [selectedResourceList, setSelectedResourceList] = useState<any[]>(initialValues?.resource_list || selectedResourceNames);
 const [selectedCondition, setSelectedCondition] = useState(initialValues?.condition || "");
 const [threshold, setThreshold] = useState<number | "">(initialValues?.value_threshold || "");
 const [date, setDate] = useState<Date | undefined>(initialValues?.ends_on ? new Date(initialValues.ends_on) : addDays(new Date(), 7));
 const [percentageThreshold, setPercentageThreshold] = useState<number | "">(initialValues?.percentage_threshold || "");

 useEffect(() => {
  if (JSON.stringify(selectedResourceList) !== JSON.stringify(selectedResourceNames)) {
    setSelectedResourceList(selectedResourceNames);
  }
}, [selectedResourceNames]);

useEffect(() => {
  setSelectedOperation("");
  setSelectedCondition("");
  setThreshold("");
  setSelectedFrequency("");

  if (alertType === "Spike") {
    setSelectedOperation("AVERAGE");
    setSelectedCondition("Greater than");
  }
}, [alertType]);

const isPercentageThreshold = alertType === "Spike"; 

 const handleRecipientSelect = (recipient: string) => {
   setSelectedRecipient(recipient);
   setSelectedIntegration(null);
   setIntegrations([]);
 };


 const handleFrequencySelect = (frequency: string) => {
   setSelectedFrequency(frequency);
 };


 const handleOperationSelect = (operation: string) => {
   setSelectedOperation(operation);
 };


 const handleConditionSelect = (condition: string) => {
   setSelectedCondition(condition);
 };

 const typeOptions = [
  { value: "Connection", label: "Connection Based" },
  { value: "Resource", label: "Resource Based" },
  { value: "Tag", label: "Tag Based" }
];

 useEffect(() => {
   const fetchIntegrations = async () => {
     if (!selectedRecipient) return;


     try {
       setLoading(true);
       const response = await axiosInstance.get(`${BACKEND}/alert/fetch-integrations`, {
         params: { recipient: selectedRecipient },
       });


       const validIntegrations = response.data.filter(
         (item: any): item is Integration =>
           typeof item === "object" &&
           item !== null &&
           typeof item.id === "number" &&
           typeof item.name === "string"
       );


       setIntegrations(validIntegrations);
     } catch (error) {
       console.error("Error fetching integrations:", error);
       toast({
         title: "Error",
         description: "Failed to fetch integrations",
         variant: "destructive",
       });
     } finally {
       setLoading(false);
     }
   };
   fetchIntegrations();
 }, [selectedRecipient, toast]);


 const handleIntegrationSelect = (integration: Integration) => {
   setSelectedIntegration(integration);
 };


 const validateForm = (): boolean => {
  const config = alertTypeConfigs[alertType as keyof typeof alertTypeConfigs];
  if (!config) return false;

  if (!alertName) {
    toast({
      title: "Error",
      description: "Please enter an alert name",
      variant: "destructive",
    });
    return false;
  }

  for (const field of config.requiredFields) {
    if (field === 'operation' && !selectedOperation) {
      toast({
        title: "Error",
        description: "Please select an operation",
        variant: "destructive",
      });
      return false;
    }
    if (field === 'condition' && !selectedCondition) {
      toast({
        title: "Error",
        description: "Please select a condition",
        variant: "destructive",
      });
      return false;
    }
    if (field === 'threshold') {
      // Only validate threshold based on alert type
      if (alertType === "Cost" && threshold === "") {
        toast({
          title: "Error",
          description: "Please enter a value threshold for the Cost alert.",
          variant: "destructive",
        });
        return false;
      }
      if (alertType === "Spike" && threshold === "" && percentageThreshold === "") {
        toast({
          title: "Error",
          description: "Please enter either a value threshold or a percentage threshold for the Spike alert.",
          variant: "destructive",
        });
        return false;
      }
    }
    if (field === 'schedule' && !selectedFrequency) {
      toast({
        title: "Error",
        description: "Please select a schedule frequency",
        variant: "destructive",
      });
      return false;
    }
    if (field === 'recipient' && !selectedRecipient) {
      toast({
        title: "Error",
        description: "Please select a recipient",
        variant: "destructive",
      });
      return false;
    }
  }
  return true;
};

const handleSubmit = async () => {
  if (!validateForm()) {
    return;
  }

  const normalizedTagIds = selectedTagIds.map(id =>
    typeof id === 'string' ? parseInt(id, 10) : id
  ).filter(id => !isNaN(id));

  const config = alertTypeConfigs[alertType as keyof typeof alertTypeConfigs];

  // Determine which threshold to use based on the alert type
  let thresholdValue = null;
  let percentageThresholdValue = null;

  if (alertType === "Spike") {
    if (percentageThreshold !== "" && threshold !== "") {
      toast({
        title: "Error",
        description: "Please provide either a percentage threshold or a value threshold, not both.",
        variant: "destructive",
      });
      return;
    }
    percentageThresholdValue = percentageThreshold !== "" ? Number(percentageThreshold) : null;
    thresholdValue = threshold !== "" ? Number(threshold) : null;
  } else if (alertType === "Cost") {
    if (threshold === "") {
      toast({
        title: "Error",
        description: "Please enter a value threshold for the Cost alert.",
        variant: "destructive",
      });
      return;
    }
    thresholdValue = Number(threshold);
  }

  const alertRuleData: AlertRuleData = {
    name: alertName,
    recipient: selectedRecipient,
    integration_id: selectedIntegration?.id,
    ends_on: date ? format(date, 'yyyy-MM-dd') : format(addDays(new Date(), 7), 'yyyy-MM-dd'),
    schedule: selectedFrequency,
    status: alertStatus,
    type: selectedType,
    alert_type: alertType,
    resource_list: selectedResourceList,
    percentage_threshold: percentageThresholdValue,
    value_threshold: thresholdValue,
    condition: selectedCondition,
    operation: selectedOperation,
    state: {},
    tag_ids: normalizedTagIds,
    project_ids: [parseInt(projectId)]
  };

  try {
    const data = await saveAlertRule(alertRuleData);
    console.log("Alert rule saved:", data);
    toast({
      title: "Success",
      description: "Alert rule saved successfully",
    });
    onSave();
    window.location.href = '/alerts/alertRules';
  } catch (error) {
    console.error("Error saving alert rule:", error);
    toast({
      title: "Error",
      description: "Failed to save alert rule",
      variant: "destructive",
    });
  }
};

const disableDate = (date: Date) => date < new Date(new Date().setHours(0, 0, 0, 0));

 return (
  <div>
  <h2 className="text-2xl mb-4 mt-8">Create New Alert Rule</h2>
  
  {/* Type and Alert Type Selection */}
  <div className="flex gap-4 mb-4">
    <div className="flex gap-4 items-center">
      <span>Type: </span>
      <Menu as="div" className="relative inline-block text-left">
        <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
          {selectedType || "Connection Based"}
          <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
        </MenuButton>
        <MenuItems className="absolute left-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
          {typeOptions.map((type) => (
            <MenuItem key={type.value}>
              {({ active }) => (
                <button
                  className={`${active ? "bg-blue-100" : ""} block w-full text-left px-4 py-2 text-sm ${
                    selectedType === type.value ? "bg-gray-200" : ""
                  }`}
                  onClick={() => setSelectedType(type.value)}
                >
                  {type.label}
                </button>
              )}
            </MenuItem>
          ))}
        </MenuItems>
      </Menu>
    </div>

    <div className="flex gap-4 items-center">
      <span>Alert Type: </span>
      <Menu as="div" className="relative inline-block text-left">
        <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
          {alertType || "Select Alert Type"}
          <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
        </MenuButton>
        <MenuItems className="absolute left-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
          {Object.keys(alertTypeConfigs).map((type) => (
            <MenuItem key={type}>
              {({ active }) => (
                <button
                  className={`${active ? "bg-blue-100" : ""} block w-full text-left px-4 py-2 text-sm`}
                  onClick={() => setAlertType(type)}
                >
                  {type}
                </button>
              )}
            </MenuItem>
          ))}
        </MenuItems>
      </Menu>
    </div>
  </div>

  {/* Name and Date Selection */}
  <div className="flex gap-4">
    <div className="flex gap-4 items-center">
      <span>Name </span>
      <Input className="w-72" value={alertName} onChange={(e) => setAlertName(e.target.value)} />
    </div>

    <div className="flex gap-4 items-center">
      <span className="w-24">Alert Ends on: </span>
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            className={cn(
              "w-[240px] justify-start text-left font-normal",
              !date && "text-muted-foreground"
            )}
          >
            <CalendarIcon className="mr-2 h-4 w-4" />
            {date ? format(date, "PPP") : <span>Pick a date</span>}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="start">
          <Calendar
            mode="single"
            selected={date}
            onSelect={setDate}
            disabled={disableDate}
            initialFocus
            defaultMonth={new Date()}
          />
        </PopoverContent>
      </Popover>
      <div className="flex items-center space-x-2 ml-4">
        <Switch
          id="Alert-Status"
          className="data-[state=checked]:bg-[#2F27CE] data-[state=checked]:border-[#2F27CE]"
          checked={alertStatus}
          onCheckedChange={(checked) => {
            if (alertStatus !== checked) {
              setAlertStatus(checked);
            }
          }}
        />
        <Label htmlFor="Alert-Status">Alert Status</Label>
      </div>
    </div>
  </div>

  <hr className="my-4 border-gray-300" />

  {/* Alert Conditions */}
  <div className="flex gap-4">
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-4 mt-8">
        <div>Define alert condition:</div>
        <div className="flex gap-4 ml-8 items-center">
          {/* Operation Dropdown */}
          {alertTypeConfigs[alertType as keyof typeof alertTypeConfigs]?.requiredFields.includes('operation') && alertType !== "Spike" && (
            <Menu as="div" className="relative inline-block text-left">
              <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
                {selectedOperation || "Select Operation"}
                <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
              </MenuButton>
              <MenuItems className="absolute left-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
                {alertTypeConfigs[alertType as keyof typeof alertTypeConfigs]?.operations.map((operation) => (
                  <MenuItem key={operation}>
                    {({ active }) => (
                      <button
                        className={`${active ? "bg-blue-100" : ""} block w-full text-left px-4 py-2 text-sm ${selectedOperation === operation ? "bg-gray-200" : ""}`}
                        onClick={() => handleOperationSelect(operation)}
                      >
                        {operation}
                      </button>
                    )}
                  </MenuItem>
                ))}
              </MenuItems>
            </Menu>
          )}

          {/* Condition Dropdown */}
          {alertTypeConfigs[alertType as keyof typeof alertTypeConfigs]?.requiredFields.includes('condition') && alertType !== "Spike" && (
            <Menu as="div" className="relative inline-block text-left">
              <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
                {selectedCondition || "Select Condition"}
                <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
              </MenuButton>
              <MenuItems className="absolute left-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
                {alertTypeConfigs[alertType as keyof typeof alertTypeConfigs]?.conditions.map((condition) => (
                  <MenuItem key={condition.value}>
                    {({ active }) => (
                      <button
                        className={`${active ? "bg-blue-100" : ""} block w-full text-left px-4 py-2 text-sm ${selectedCondition === condition.value ? "bg-gray-200" : ""}`}
                        onClick={() => handleConditionSelect(condition.value)}
                      >
                        {condition.label}
                      </button>
                    )}
                  </MenuItem>
                ))}
              </MenuItems>
            </Menu>
          )}

          {/* Threshold Inputs */}
          <div className="flex items-center gap-2">
            {/* Value Threshold */}
            <div className="flex items-center">
              <Input
                className="w-37"
                placeholder="Enter Value threshold"
                value={threshold}
                onChange={(e) => {
                  if (alertType !== "Spike" || !percentageThreshold) {
                    setThreshold(e.target.value === "" ? "" : parseFloat(e.target.value));
                  }
                }}
              />
              <span className="ml-2 text-gray-600 font-medium">$</span>
            </div>

            {/* Percentage Threshold (only for Spike alerts) */}
            {alertType === "Spike" && (
              <div className="flex items-center">
                <Input
                  className="w-37"
                  placeholder="Enter Percentage threshold"
                  value={percentageThreshold}
                  onChange={(e) => {
                    if (!threshold) {
                      setPercentageThreshold(e.target.value === "" ? "" : parseFloat(e.target.value));
                    }
                  }}
                />
                <span className="ml-2 text-gray-600 font-medium">%</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Schedule Selection */}
      <div className="mt-8 flex gap-4 items-center">
        <span>Schedule: </span>
        <Menu as="div" className="relative inline-block text-left">
          <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
            {selectedFrequency || "Select Schedule"}
            <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
          </MenuButton>
          <MenuItems className="absolute left-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
            {alertTypeConfigs[alertType as keyof typeof alertTypeConfigs]?.scheduleOptions.map((schedule) => (
              <MenuItem key={schedule}>
                {({ active }) => (
                  <button
                    className={`${active ? "bg-blue-100" : ""} block w-full text-left px-4 py-2 text-sm`}
                    onClick={() => handleFrequencySelect(schedule)}
                  >
                    {schedule}
                  </button>
                )}
              </MenuItem>
            ))}
          </MenuItems>
        </Menu>
      </div>

      {/* Recipients Section */}
      <div className="mt-8">
        <span className="mr-4">Recipients:</span>
        <Menu as="div" className="relative inline-block text-left">
          <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
            {selectedRecipient || "Select Recipients"}
            <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
          </MenuButton>
          <MenuItems className="absolute left-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
            {loading ? (
              <div className="flex justify-center py-2">
                <Loader className="animate-spin" />
              </div>
            ) : (
              <>
                <MenuItem>
                  {({ active }) => (
                    <button
                      className={`${active ? "bg-blue-100" : ""} block w-full text-left px-4 py-2 text-sm`}
                      onClick={() => handleRecipientSelect("Slack")}
                    >
                      Slack
                    </button>
                  )}
                </MenuItem>
                <MenuItem>
                  {({ active }) => (
                    <button
                      className={`${active ? "bg-blue-100" : ""} block w-full text-left px-4 py-2 text-sm`}
                      onClick={() => handleRecipientSelect("microsoft_teams")}
                    >
                      Microsoft Teams
                    </button>
                  )}
                </MenuItem>
              </>
            )}
          </MenuItems>
        </Menu>
      </div>
    </div>
  </div>

  {/* Integrations Section */}
  {selectedRecipient && (
    <div className="mt-4">
      <span className="mr-4">Integrations:</span>
      <Menu as="div" className="relative inline-block text-left">
        <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
          {selectedIntegration ? selectedIntegration.name : "Select Integration"}
          <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
        </MenuButton>
        <MenuItems className="absolute left-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
          {loading ? (
            <div className="flex justify-center py-2">
              <Loader className="animate-spin" />
            </div>
          ) : integrations.length > 0 ? (
            integrations.map((integration) => (
              <MenuItem key={integration.id}>
                {({ active }) => (
                  <button
                    className={`${active ? "bg-blue-100" : ""} block w-full text-left px-4 py-2 text-sm`}
                    onClick={() => handleIntegrationSelect(integration)}
                  >
                    {integration.name}
                  </button>
                )}
              </MenuItem>
            ))
          ) : (
            <div className="py-2 px-4 text-gray-500">No integrations found</div>
          )}
        </MenuItems>
      </Menu>
    </div>
  )}

  <hr className="my-4 border-gray-300 w-[80%]" />
  
  {/* Action Buttons */}
  <div className="mt-4">
    <Button onClick={onBack} variant="outline" className="mr-2 border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10 text-sm font-medium rounded-md">
      Back
    </Button>
    <Button variant="default" onClick={handleSubmit} className="bg-[#D82026] text-white rounded-md hover:bg-[#b81a1f] text-sm font-medium h-10 px-4 py-2">
      Save Alert Rule
    </Button>
  </div>
</div>
   
 );
};


export default NewAlertRuleForm;

