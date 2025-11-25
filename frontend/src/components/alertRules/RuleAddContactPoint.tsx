import React, { useState } from "react";
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
import { ChevronDownIcon, Loader, Unplug } from "lucide-react";
import { testSlackConnection, testTeamsConnection, saveIntegration } from "@/lib/api";

interface Integration {
  id: number;
  type: "slack" | "microsoft_teams" | "email" | null;
  webhookUrl?: string;
  text?: string;
  email?: string;
  username?: string;
  iconEmoji?: string;
}

interface RuleAddContactPointProps {
  onBack: () => void;
  onSave: () => void;
  projectId: string;
}

const RuleAddContactPoint: React.FC<RuleAddContactPointProps> = ({
  onBack,
  onSave,
  projectId,
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
  const [canSave, setCanSave] = useState(false);

  const form = useForm({
    defaultValues: {
      name: "",
      webhookUrl: "",
      text: "",
      email: "",
      username: "",
      iconEmoji: "",
    },
  });

  const validateWebhookUrl = (url: string, type: "slack" | "microsoft_teams" | "email"): boolean => {
    if (type === "email") return /.+@.+\..+/.test(url);
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

  const handleIntegrationSelect = (id: number, integrationType: "slack" | "microsoft_teams" | "email") => {
    setIntegrations((prev) =>
      prev.map((integration) =>
        integration.id === id ? { ...integration, type: integrationType } : integration
      )
    );
    setTestResult(null);
    setCanSave(false);
  };

  const handleTestConnection = async (id: number) => {
    const integration = integrations.find((i) => i.id === id);
    const webhookUrl = form.getValues("webhookUrl");
    const email = form.getValues("email");

    if (!integration) return;
    if (integration.type === "email") {
      if (!validateWebhookUrl(email, "email")) {
        setTestResult({ id, success: false, message: "Invalid email address" });
        setCanSave(false);
        return;
      }
      setTestResult({ id, success: true, message: "Valid email" });
      setCanSave(form.getValues("name").trim() !== "");
      return;
    }

    if (!webhookUrl || !validateWebhookUrl(webhookUrl, integration.type!)) {
      setTestResult({
        id,
        success: false,
        message: `Invalid ${integration.type} webhook URL format`,
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
    } catch {
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
    if (!canSave) return;

    const integration = integrations[0];
    if (integration && integration.type) {
      try {
        await saveIntegration({
          name: data.name,
          integration_type: integration.type,
          url: integration.type === "email" ? data.email : data.webhookUrl,
          notification_template: {
            text: data.text,
            username: "WebhookBot",
            icon_emoji: integration.type === "slack" ? "robot_face" : undefined,
          },
          project_id: parseInt(projectId),
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
    }
  };

  const IntegrationSection = ({ integration }: { integration: Integration }) => (
    <div key={integration.id}>
      <hr className="m-4 border-[#E0E5EF]" />
      <div className="flex justify-between">
        <Menu as="div" className="relative inline-block text-left">
          <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
            {integration.type ? (integration.type === "slack" ? "Slack" : integration.type === "email" ? "Email" : "Microsoft Teams") : "Select Integration"}
            <ChevronDownIcon className="-mr-1 h-5 w-5 text-[#6B7280]" aria-hidden="true" />
          </MenuButton>
          <MenuItems className="absolute left-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
            <MenuItem as="button" className="block w-full text-left px-4 py-2 text-sm text-[#233E7D] hover:bg-[#233E7D]/10" onClick={() => handleIntegrationSelect(integration.id, "slack")}>Slack</MenuItem>
            <MenuItem as="button" className="block w-full text-left px-4 py-2 text-sm text-[#233E7D] hover:bg-[#233E7D]/10" onClick={() => handleIntegrationSelect(integration.id, "microsoft_teams")}>Microsoft Teams</MenuItem>
            <MenuItem as="button" className="block w-full text-left px-4 py-2 text-sm text-[#233E7D] hover:bg-[#233E7D]/10" onClick={() => handleIntegrationSelect(integration.id, "email")}>Email</MenuItem>
          </MenuItems>
        </Menu>
        <Button
          variant="outline"
          className="mr-2 border-[#233E7D] text-[#233E7D] flex justify-center items-center gap-2 text-sm font-medium hover:bg-[#233E7D]/10"
          onClick={() => handleTestConnection(integration.id)}
          disabled={testingConnection === integration.id || !integration.type}
        >
          {testingConnection === integration.id ? <Loader size={16} className="animate-spin" /> : <Unplug size={16} />} Test
        </Button>
      </div>
      {(integration.type === "slack" || integration.type === "microsoft_teams") && (
        <div className="mt-4 ml-4">
          <FormField
            control={form.control}
            name="webhookUrl"
            render={({ field }) => (
              <FormItem>
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
      {integration.type === "email" && (
        <div className="mt-4 ml-4">
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Input
                    type="email"
                    placeholder="Enter email address"
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
        <div className={`mt-2 ${testResult.success ? "text-[#22C55E]" : "text-[#D82026]"} text-sm`}>{testResult.message}</div>
      )}
    </div>
  );

  return (
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
                  <Input placeholder="Name" className="mt-2 w-96 border-[#C8C8C8] rounded-md px-3 py-2 text-base" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          {integrations.map((integration) => (
            <IntegrationSection key={integration.id} integration={integration} />
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
  );
};

export default RuleAddContactPoint;
