// src/lib/dateUtils.ts

import { DateRangePreset } from "@/types/recommendations";
import { subDays, subMonths, subYears, startOfDay, startOfWeek, startOfMonth, startOfYear } from "date-fns";

export interface DateRange {
    startDate: Date;
    endDate: Date;
}

/**
 * Calculate date range based on preset option
 * All ranges are calculated relative to current date
 */
export function calculateDateRange(preset: DateRangePreset): DateRange | null {
    const now = new Date();
    const today = startOfDay(now);

    switch (preset) {
        case 'today':
            return {
                startDate: today,
                endDate: today
            };

        case 'yesterday':
            const yesterday = subDays(today, 1);
            return {
                startDate: yesterday,
                endDate: yesterday
            };

        case 'last_week':
            return {
                startDate: subDays(today, 7),
                endDate: today
            };

        case 'last_month':
            return {
                startDate: subMonths(today, 1),
                endDate: today
            };

        case 'last_6_months':
            return {
                startDate: subMonths(today, 6),
                endDate: today
            };

        case 'last_year':
            return {
                startDate: subYears(today, 1),
                endDate: today
            };

        case 'custom':
            // Return null for custom - user will set dates manually
            return null;

        default:
            return null;
    }
}
