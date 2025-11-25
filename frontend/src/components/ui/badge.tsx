import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-cp-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-cp-blue focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-cp-blue text-white hover:bg-cp-blue-hover",
        secondary:
          "border-transparent bg-cp-red text-white hover:bg-cp-red-hover",
        destructive:
          "border-transparent bg-cp-red text-white hover:bg-cp-red-hover",
        outline: "text-cp-blue border-cp-blue",
        custom:
          "border-transparent text-cp-red",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
