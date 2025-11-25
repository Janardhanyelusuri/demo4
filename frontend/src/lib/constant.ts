import { LayoutDashboard } from "lucide-react";
import { CloudCog } from "lucide-react";
import { Settings, Bell } from "lucide-react";

export const menuOptions = [
  {
    name: "Notification",
    Component: CloudCog,
    href: ["/connections"],
  },
  { name: "Home", Component: LayoutDashboard, href: ["/dashboard-home"] },
  { name: "Alerts", href: "/alerts/alertRules", Component: Bell },
  { name: "Settings", Component: Settings, href: "/settings" }
];
