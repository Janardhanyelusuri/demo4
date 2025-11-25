// src/components/recommendations/RecommendationCard.tsx
import React from 'react';
import { NormalizedRecommendation } from "@/types/recommendations";
import { AlertCircle, Clock, CheckCircle, Info } from 'lucide-react';

// Import the specialized card components
import KPICard from './KPICard';
import MetricsCard from './MetricsCard';
import EffectiveRecommendationCard from './EffectiveRecommendationCard';
import AdditionalRecommendationsCard from './AdditionalRecommendationsCard';
import AnomaliesCard from './AnomaliesCard';
import ContractDealCard from './ContractDealCard';

interface RecommendationCardProps {
    recommendation: NormalizedRecommendation;
}

const getSeverityStyles = (severity: 'High' | 'Medium' | 'Low') => {
    switch (severity) {
        case 'High':
            return {
                icon: <AlertCircle className="w-6 h-6 text-red-500" />,
                bg: 'from-red-50 to-orange-50',
                border: 'border-red-300',
                badge: 'bg-red-100 text-red-800',
                color: 'text-red-700'
            };
        case 'Medium':
            return {
                icon: <Clock className="w-6 h-6 text-yellow-500" />,
                bg: 'from-yellow-50 to-amber-50',
                border: 'border-yellow-300',
                badge: 'bg-yellow-100 text-yellow-800',
                color: 'text-yellow-700'
            };
        case 'Low':
            return {
                icon: <CheckCircle className="w-6 h-6 text-green-500" />,
                bg: 'from-green-50 to-emerald-50',
                border: 'border-green-300',
                badge: 'bg-green-100 text-green-800',
                color: 'text-green-700'
            };
        default:
            return {
                icon: <Info className="w-6 h-6 text-gray-500" />,
                bg: 'from-gray-50 to-slate-50',
                border: 'border-gray-300',
                badge: 'bg-gray-100 text-gray-800',
                color: 'text-gray-700'
            };
    }
};

const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation: rec }) => {
    const styles = getSeverityStyles(rec.severity);
    // Show last 3 parts of resource ID for better context (e.g., storageaccount/blobServices/default)
    const parts = rec.resourceId.split('/');
    const resourceIdShort = parts.length > 3
        ? parts.slice(-3).join('/')
        : rec.resourceId;

    return (
        <div className="space-y-4">
            {/* Header Section - More Compact */}
            <div className={`bg-gradient-to-r ${styles.bg} p-4 rounded-lg border ${styles.border} shadow-sm`}>
                <div className="flex justify-between items-center gap-4">
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                        <div className="flex-shrink-0">{styles.icon}</div>
                        <div className="min-w-0 flex-1">
                            <h2 className="text-xl font-bold text-gray-900">Resource Analysis</h2>
                            <p className="text-xs text-gray-600 mt-0.5 truncate">
                                <span className="font-semibold">Resource:</span> {resourceIdShort}
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-4 flex-shrink-0">
                        <span className={`${styles.badge} px-3 py-1 rounded-full text-xs font-bold`}>
                            {rec.severity}
                        </span>
                        <div className="text-right">
                            <p className="text-[10px] text-gray-500 uppercase font-semibold">Savings</p>
                            <p className="text-2xl font-bold text-green-600">{rec.totalSavingPercent}%</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* KPI Cards Section - More Compact */}
            <div>
                <h3 className="text-sm font-bold text-gray-700 mb-2 uppercase tracking-wide">Key Metrics</h3>
                <KPICard recommendation={rec} />
            </div>

            {/* Main Content Grid - Reduced Spacing */}
            <div className="space-y-4">
                {/* Metrics Card - Full Width */}
                <MetricsCard recommendation={rec} />

                {/* Recommendations Row - Cards stretch to equal height */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-stretch">
                    <EffectiveRecommendationCard recommendation={rec} />
                    <AdditionalRecommendationsCard recommendation={rec} />
                </div>

                {/* Collapsible Sections Row - Cards stretch to equal height */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-stretch">
                    <AnomaliesCard recommendation={rec} />
                    <ContractDealCard recommendation={rec} />
                </div>
            </div>
        </div>
    );
};

export default RecommendationCard;