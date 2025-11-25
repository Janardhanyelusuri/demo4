import React, { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormMessage,
} from "@/components/ui/form";
import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import { ChevronDownIcon, Loader, Unplug, X } from "lucide-react";
import { testSlackConnection, testTeamsConnection, saveIntegration, fetchProjects } from "@/lib/api";
import { Dialog, Transition } from "@headlessui/react";

interface Integration {
  id: number;
  type: "slack" | "microsoft_teams" | null;
  webhookUrl?: string;
  text?: string;
  username?: string;
  iconEmoji?: string;
}

interface CloudProvider {
  id: number;
  name: string;
}

interface Project {
  id: string;
  name: string;
  cloud_platform: "aws" | "gcp" | "azure" | "snowflake";
}

interface RuleAddContactPointProps {
  onBack: () => void;
  onSave: () => void;
}

const RuleAddContactPoint: React.FC<RuleAddContactPointProps> = ({
  onBack,
  onSave,
}) => {
  const [integrations, setIntegrations] = useState<Integration[]>([
    { id: 1, type: null },
  ]);
  const [testingConnection, setTestingConnection] = useState<number | null>(null);
  const [testResult, setTestResult] = useState<{
    id: number;
    success: boolean;
    message?: string;
  } | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(true);
  const [cloudProviders] = useState<CloudProvider[]>([
    { id: 1, name: "AWS" },
    { id: 2, name: "GCP" },
    { id: 3, name: "Azure" },
  ]);
  const [selectedCloudProvider, setSelectedCloudProvider] = useState<CloudProvider | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [filteredProjects, setFilteredProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [canSave, setCanSave] = useState(false);

  const form = useForm({
    defaultValues: {
      name: "",
      webhookUrl: "",
      text: "",
      username: "",
      iconEmoji: "",
    },
  });

  useEffect(() => {
    const getProjects = async () => {
      try {
        const fetchedProjects = await fetchProjects();
        setProjects(fetchedProjects);
      } catch (error) {
        console.error("Error fetching projects:", error);
      }
    };

    getProjects();
  }, []);

  const validateWebhookUrl = (url: string, type: "slack" | "microsoft_teams"): boolean => {
    try {
      const urlObj = new URL(url);
      if (type === "slack") {
        return urlObj.hostname === "hooks.slack.com" && urlObj.pathname.startsWith("/services/");
      } else if (type === "microsoft_teams") {
        return urlObj.hostname === "outlook.office.com" || 
               urlObj.hostname.endsWith(".webhook.office.com");
      }
      return false;
    } catch {
      return false;
    }
  };

  const handleCloudSelect = (cloud: CloudProvider) => {
    setSelectedCloudProvider(cloud);
    setSelectedProject(null);
    const filtered = projects.filter((project) => project.cloud_platform === cloud.name.toLowerCase());
    setFilteredProjects(filtered);
  };

  const handleProjectSelect = (project: Project) => {
    setSelectedProject(project);
  };

  const handleProceedToForm = () => {
    if (!selectedProject) return;
    setIsDialogOpen(false);
  };

  const handleCloseDialog = () => {
    setIsDialogOpen(false);
    onBack();
  };

  const handleIntegrationSelect = (id: number, integrationType: "slack" | "microsoft_teams") => {
    setIntegrations((prevIntegrations) =>
      prevIntegrations.map((integration) =>
        integration.id === id
          ? { ...integration, type: integrationType }
          : integration
      )
    );
    setTestResult(null);
    setCanSave(false);
  };

  const handleTestConnection = async (id: number) => {
    const integration = integrations.find((i) => i.id === id);
    const webhookUrl = form.getValues("webhookUrl");

    if (!integration || !webhookUrl) {
      setTestResult({
        id,
        success: false,
        message: "Please provide a webhook URL"
      });
      setCanSave(false);
      return;
    }

    if (!validateWebhookUrl(webhookUrl, integration.type!)) {
      setTestResult({
        id,
        success: false,
        message: `Invalid ${integration.type === "slack" ? "Slack" : "Teams"} webhook URL format`
      });
      setCanSave(false);
      return;
    }

    setTestingConnection(id);
    setTestResult(null);

    try {
      let success = false;
      if (integration.type === "slack") {
        success = await testSlackConnection(webhookUrl);
      } else if (integration.type === "microsoft_teams") {
        success = await testTeamsConnection(webhookUrl);
      }

      setTestResult({
        id,
        success,
        message: success ? "Connection successful!" : "Connection failed. Please check your webhook URL."
      });
      setCanSave(success && form.getValues("name").trim() !== "");
    } catch (error) {
      setTestResult({
        id,
        success: false,
        message: "Connection test failed. Please try again."
      });
      setCanSave(false);
    } finally {
      setTestingConnection(null);
    }
  };

  const onSubmit = async (data: any) => {
    if (!canSave || !selectedProject) return;
    const integration = integrations[0];
    if (!integration || !integration.type) return;

    try {
      await saveIntegration({
        name: data.name,
        integration_type: integration.type,
        url: data.webhookUrl,
        notification_template: {
          text: data.text,
          username: "WebhookBot",
          icon_emoji: integration.type === "slack" ? "robot_face" : undefined,
        },
        project_id: parseInt(selectedProject.id),
      });
      onSave();
    } catch (error) {
      console.error("Error saving integration:", error);
      setTestResult({
        id: integration.id,
        success: false,
        message: "Failed to save integration. Please try again."
      });
    }
  };

  const IntegrationSection = ({
    integration,
  }: {
    integration: Integration;
  }) => (
    <div key={integration.id}>
      <hr className="m-4 border-[#E0E5EF]" />
      <div className="flex justify-between">
        <Menu as="div" className="relative inline-block text-left">
          <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
            {integration.type ? 
              (integration.type === "slack" ? "Slack" : "Microsoft Teams") : 
              "Select Integration"}
            <ChevronDownIcon className="-mr-1 h-5 w-5 text-[#6B7280]" aria-hidden="true" />
          </MenuButton>
          <MenuItems className="absolute left-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
            <MenuItem
              as="button"
              className="block w-full text-left px-4 py-2 text-sm text-[#233E7D] hover:bg-[#233E7D]/10"
              onClick={() => handleIntegrationSelect(integration.id, "slack")}
            >
              Slack
            </MenuItem>
            <MenuItem
              as="button"
              className="block w-full text-left px-4 py-2 text-sm text-[#233E7D] hover:bg-[#233E7D]/10"
              onClick={() => handleIntegrationSelect(integration.id, "microsoft_teams")}
            >
              Microsoft Teams
            </MenuItem>
          </MenuItems>
        </Menu>
        <Button
          variant="outline"
          className="mr-2 border-[#233E7D] text-[#233E7D] flex justify-center items-center gap-2 text-sm font-medium hover:bg-[#233E7D]/10"
          onClick={() => handleTestConnection(integration.id)}
          disabled={testingConnection === integration.id || !integration.type}
        >
          {testingConnection === integration.id ? (
            <Loader size={16} className="animate-spin" />
          ) : (
            <Unplug size={16} />
          )}
          Test
        </Button>
      </div>
      {integration.type && (
        <div className="mt-4 ml-4">
          <FormField
            control={form.control}
            name="webhookUrl"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="text-[#233E7D] text-sm font-semibold">Webhook URL</FormLabel>
                <FormControl>
                  <Input
                    placeholder={`Enter ${integration.type === "slack" ? "Slack" : "Teams"} webhook URL`}
                    className="mt-2 w-96 border-[#C8C8C8] rounded-md px-3 py-2 text-base"
                    {...field}
                    onChange={(e) => {
                      field.onChange(e);
                      setTestResult(null);
                      setCanSave(false);
                    }}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
      )}
      {testResult && testResult.id === integration.id && (
        <div
          className={`mt-2 ${
            testResult.success ? "text-[#22C55E]" : "text-[#D82026]"
          } text-sm`}
        >
          {testResult.message}
        </div>
      )}
    </div>
  );

  return (
    <>
      <Transition show={isDialogOpen}>
        <Dialog onClose={handleCloseDialog} className="relative z-50">
          <Transition.Child>
            <div className="fixed inset-0 bg-black bg-opacity-25"></div>
          </Transition.Child>
          <div className="fixed inset-0 flex items-center justify-center p-4">
            <Transition.Child>
              <Dialog.Panel className="w-full max-w-md p-6 bg-white rounded-md relative border border-[#E0E5EF]">
                <button
                  onClick={handleCloseDialog}
                  className="absolute top-4 right-4 p-1 rounded-full hover:bg-[#F9FEFF]"
                >
                  <X className="h-5 w-5 text-[#6B7280]" />
                </button>
                <Dialog.Title className="text-xl font-semibold text-[#233E7D]">Select Cloud Provider</Dialog.Title>
                <Menu as="div" className="relative inline-block text-left mt-4">
                  <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
                    {selectedCloudProvider ? selectedCloudProvider.name : "Select Cloud Provider"}
                    <ChevronDownIcon className="-mr-1 h-5 w-5 text-[#6B7280]" aria-hidden="true" />
                  </MenuButton>
                  <MenuItems className="absolute z-10 mt-2 w-full origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
                    {cloudProviders.map((cloud) => (
                      <MenuItem key={cloud.id}>
                        {({ active }) => (
                          <button
                            className={`${
                              active ? "bg-[#233E7D]/10" : ""
                            } block w-full text-left px-4 py-2 text-sm text-[#233E7D]`}
                            onClick={() => handleCloudSelect(cloud)}
                          >
                            {cloud.name}
                          </button>
                        )}
                      </MenuItem>
                    ))}
                  </MenuItems>
                </Menu>
                {selectedCloudProvider && (
                  <>
                    <Dialog.Title className="text-xl font-semibold mt-4 text-[#233E7D]">Select Project</Dialog.Title>
                    <Menu as="div" className="relative inline-block text-left mt-4">
                      <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
                        {selectedProject ? selectedProject.name : "Select Project"}
                        <ChevronDownIcon className="-mr-1 h-5 w-5 text-[#6B7280]" aria-hidden="true" />
                      </MenuButton>
                      <MenuItems className="absolute z-10 mt-2 w-full origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none max-w-full">
                        {filteredProjects.length === 0 ? (
                          <div className="px-4 py-2 text-sm text-[#6B7280]">No projects available for selected cloud</div>
                        ) : (
                          filteredProjects.map((project) => (
                            <MenuItem key={project.id}>
                              {({ active }) => (
                                <button
                                  className={`${
                                    active ? "bg-[#233E7D]/10" : ""
                                  } block w-full text-left px-4 py-2 text-sm text-[#233E7D] overflow-hidden text-ellipsis whitespace-nowrap`}
                                  style={{ maxWidth: '100%' }}
                                  onClick={() => handleProjectSelect(project)}
                                >
                                  {project.name}
                                </button>
                              )}
                            </MenuItem>
                          ))
                        )}
                      </MenuItems>
                    </Menu>
                  </>
                )}
                <div className="mt-4">
                  <Button 
                    variant="default"
                    className="bg-[#D82026] text-white rounded-md hover:bg-[#b81a1f] text-sm font-medium h-10 px-4 py-2"
                    onClick={handleProceedToForm}
                    disabled={!selectedProject}
                  >
                    Proceed to Form
                  </Button>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </Dialog>
      </Transition>
      {!isDialogOpen && selectedProject && (
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <div className="mt-4 ml-4">
              <h1 className="text-2xl font-bold text-[#233E7D] mb-6">Create Contact Points</h1>
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-[#233E7D] text-sm font-semibold">Name</FormLabel>
                    <FormControl>
                      <Input
                        placeholder="Name"
                        className="mt-2 w-96 border-[#C8C8C8] rounded-md px-3 py-2 text-base"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              {integrations.map((integration) => (
                <IntegrationSection
                  key={integration.id}
                  integration={integration}
                />
              ))}
              <div className="mt-4">
                <Button onClick={onBack} variant="outline" className="mr-2 border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10 text-sm font-medium rounded-md">
                  Back
                </Button>
                <Button variant="default" type="submit" disabled={!canSave} className="bg-[#D82026] text-white rounded-md hover:bg-[#b81a1f] text-sm font-medium h-10 px-4 py-2">
                  Save Contact point
                </Button>
              </div>
            </div>
          </form>
        </Form>
      )}
    </>
  );
};

export default RuleAddContactPoint;

