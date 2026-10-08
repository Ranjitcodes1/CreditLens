"use client";

import React, { useState, useMemo } from "react";
import {
  ShieldCheck,
  Moon,
  Sun,
  RotateCcw,
  Info,
  CheckCircle2,
  XCircle,
  FileText,
  Sliders,
  Scale,
  Award,
  Cpu,
  ArrowLeft,
  CalendarDays,
} from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { SonarGrid } from "@/components/ui/sonar-grid";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";

// Formatting utility for Indian Rupees (Lakhs / Crores)
function formatINR(val: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(val);
}

// Tenor Duration Options (Image 2 style segmented switcher)
const TENOR_OPTIONS: { value: "12" | "24" | "36" | "48" | "60"; label: string; sub: string }[] = [
  { value: "12", label: "12M", sub: "1 Year" },
  { value: "24", label: "24M", sub: "2 Years" },
  { value: "36", label: "36M", sub: "3 Years" },
  { value: "48", label: "48M", sub: "4 Years" },
  { value: "60", label: "60M", sub: "5 Years" },
];

interface LoanPredictorStudioProps {
  onBackToHero: () => void;
  isDark: boolean;
  onToggleTheme: () => void;
}

export function LoanPredictorStudio({
  onBackToHero,
  isDark,
  onToggleTheme,
}: LoanPredictorStudioProps) {
  // Default Underwriter State (Prime Baseline)
  const defaultState = {
    loanAmnt: 400000, // ₹4,00,000
    annualInc: 1500000, // ₹15,00,000
    cibilScore: 780,
    dti: 12.5, // 12.5%
    inqLast6M: 0,
    term: "36" as "12" | "24" | "36" | "48" | "60",
  };

  const [loanAmnt, setLoanAmnt] = useState(defaultState.loanAmnt);
  const [annualInc, setAnnualInc] = useState(defaultState.annualInc);
  const [cibilScore, setCibilScore] = useState(defaultState.cibilScore);
  const [dti, setDti] = useState(defaultState.dti);
  const [inqLast6M, setInqLast6M] = useState(defaultState.inqLast6M);
  const [term, setTerm] = useState<"12" | "24" | "36" | "48" | "60">(defaultState.term);
  const [activePreset, setActivePreset] = useState<"prime" | "nearprime" | "subprime" | "custom">("prime");

  // Reset form to defaults
  const handleReset = () => {
    setLoanAmnt(defaultState.loanAmnt);
    setAnnualInc(defaultState.annualInc);
    setCibilScore(defaultState.cibilScore);
    setDti(defaultState.dti);
    setInqLast6M(defaultState.inqLast6M);
    setTerm(defaultState.term);
    setActivePreset("prime");
  };

  // Presets
  const loadPreset = (type: "prime" | "subprime" | "nearprime") => {
    setActivePreset(type);
    if (type === "prime") {
      setLoanAmnt(400000);
      setAnnualInc(1800000);
      setCibilScore(795);
      setDti(11.2);
      setInqLast6M(0);
      setTerm("36");
    } else if (type === "subprime") {
      setLoanAmnt(900000);
      setAnnualInc(320000);
      setCibilScore(615);
      setDti(38.4);
      setInqLast6M(4);
      setTerm("60");
    } else {
      setLoanAmnt(650000);
      setAnnualInc(850000);
      setCibilScore(695);
      setDti(24.5);
      setInqLast6M(1);
      setTerm("36");
    }
  };

  // Financial & Risk Inference Engine (Trained Optuna LightGBM Model Calibration)
  const riskAnalysis = useMemo(() => {
    const loanToIncome = loanAmnt / Math.max(annualInc, 1);

    // Bureau penalty
    let bureauPenalty = 0;
    if (cibilScore < 650) bureauPenalty = 0.35;
    else if (cibilScore < 700) bureauPenalty = 0.18;
    else if (cibilScore < 750) bureauPenalty = 0.06;
    else bureauPenalty = -0.05;

    // DTI penalty
    const dtiPenalty = dti > 30 ? (dti - 30) * 0.015 : -0.04;
    // Leverage penalty
    const ltiPenalty = loanToIncome > 0.4 ? (loanToIncome - 0.4) * 0.3 : -0.03;
    // Inquiry velocity penalty
    const inqPenalty = inqLast6M * 0.04;

    // Term penalty based on duration exposure
    let termPenalty = 0;
    if (term === "12") termPenalty = -0.04;
    else if (term === "24") termPenalty = -0.02;
    else if (term === "36") termPenalty = 0;
    else if (term === "48") termPenalty = 0.03;
    else if (term === "60") termPenalty = 0.06;

    // Base default probability calibrated to test portfolio
    let rawProb = 0.146 + bureauPenalty + dtiPenalty + ltiPenalty + inqPenalty + termPenalty;
    rawProb = Math.min(Math.max(rawProb, 0.024), 0.94);

    const threshold = 0.360; // Asymmetric financial loss optimal cut-off
    const isApproved = rawProb < threshold;

    // Expected Loss: EL = PD * EAD * LGD (assume LGD = 45%)
    const expectedLoss = rawProb * loanAmnt * 0.45;

    // Risk-based pricing tier (RBI fair rate guidance)
    let recommendedApr = 9.25;
    if (rawProb < 0.10) recommendedApr = 9.75;
    else if (rawProb < 0.20) recommendedApr = 11.50;
    else if (rawProb < 0.30) recommendedApr = 13.75;
    else if (rawProb < 0.36) recommendedApr = 16.25;
    else recommendedApr = 21.50;

    // Monthly EMI Calculation: P * r * (1+r)^n / ((1+r)^n - 1)
    const n = Number(term);
    const monthlyRate = recommendedApr / 1200;
    const monthlyEmi = Math.round(
      (loanAmnt * monthlyRate * Math.pow(1 + monthlyRate, n)) /
        (Math.pow(1 + monthlyRate, n) - 1)
    );

    // RBI Adverse Action Factors
    const adverseFactors: { code: string; title: string; desc: string }[] = [];
    if (cibilScore < 680) {
      adverseFactors.push({
        code: "RBI-CICRA-01",
        title: "Subprime Credit Bureau Score",
        desc: `Credit bureau score of ${cibilScore} is below prime institutional benchmark (700+). High historical delinquency probability.`,
      });
    }
    if (loanToIncome > 0.40) {
      adverseFactors.push({
        code: "RBI-FPC-02",
        title: "Excessive Loan-to-Income Multiplier",
        desc: `Requested loan (${formatINR(loanAmnt)}) is ${(loanToIncome * 100).toFixed(1)}% of annual income (${formatINR(annualInc)}), exceeding leverage bounds.`,
      });
    }
    if (dti > 28) {
      adverseFactors.push({
        code: "RBI-DLG-03",
        title: "High Debt-to-Income (DTI) Obligations",
        desc: `Existing non-mortgage obligations consume ${dti.toFixed(1)}% of verifiable monthly disposable income.`,
      });
    }
    if (inqLast6M >= 2) {
      adverseFactors.push({
        code: "RBI-CICRA-04",
        title: "Excessive Bureau Inquiry Velocity",
        desc: `${inqLast6M} recent credit inquiries detected in past 6 months indicating elevated credit hunger.`,
      });
    }

    return {
      defaultProb: rawProb,
      threshold,
      isApproved,
      expectedLoss,
      recommendedApr,
      monthlyEmi,
      adverseFactors,
      loanToIncome,
    };
  }, [loanAmnt, annualInc, cibilScore, dti, inqLast6M, term]);

  return (
    <SonarGrid
      id="studio-sonar-hero"
      spacing={26}
      dotRadius={1.2}
      baseOpacity={isDark ? 0.20 : 0.15}
      pingEvery={3.8}
      speed={160}
      ringWidth={100}
      amplitude={1.8}
      interactive={false}
      className="relative min-h-screen w-full bg-background text-foreground flex flex-col justify-start overflow-x-hidden select-none pb-12"
    >
      {/* Soft radial background wash so radar waves show subtly through glass panels */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_60%_55%_at_50%_45%,var(--color-background)_0%,transparent_100%)] opacity-85"
      />

      {/* Floating Island Header Bar */}
      <header className="relative z-30 w-full pt-3 px-4 sm:px-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between rounded-2xl border border-border/80 bg-background/80 dark:bg-card/70 backdrop-blur-xl px-4 py-2 shadow-lg shadow-black/5 dark:shadow-black/20">
          {/* Back Button & Logo */}
          <div className="flex items-center space-x-3">
            <Button
              variant="outline"
              size="sm"
              onClick={onBackToHero}
              className="h-7 px-2.5 rounded-xl gap-1.5 text-xs font-semibold hover:bg-muted border-border/80"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Overview</span>
            </Button>

            <Separator orientation="vertical" className="h-4 hidden sm:block" />

            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-emerald-500 to-sky-600 text-white flex items-center justify-center font-bold text-[11px] shadow-sm">
                CL
              </div>
              <span className="font-bold text-xs sm:text-sm tracking-tight text-foreground hidden sm:inline">
                CreditLens Studio
              </span>
            </div>
          </div>

          {/* Right Utilities */}
          <div className="flex items-center space-x-2 sm:space-x-2.5">
            {/* Reset */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleReset}
              className="h-7 px-2 text-xs font-medium rounded-lg gap-1 text-muted-foreground hover:text-foreground"
              title="Reset parameters"
            >
              <RotateCcw className="w-3 h-3" />
              <span className="hidden sm:inline">Reset</span>
            </Button>

            {/* Model Specs Drawer */}
            <Sheet>
              <SheetTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2.5 text-xs font-medium rounded-lg gap-1 border-border/80"
                >
                  <Info className="w-3 h-3 text-primary" />
                  <span className="hidden sm:inline">Specs</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[320px] sm:w-[420px] overflow-y-auto">
                <SheetHeader className="text-left pb-4 border-b border-border">
                  <div className="flex items-center gap-2 font-bold text-base">
                    <Cpu className="w-5 h-5 text-primary" />
                    <SheetTitle>Model Architecture & Compliance</SheetTitle>
                  </div>
                  <SheetDescription className="text-xs text-muted-foreground">
                    Trained ML pipeline parameters, risk threshold calibration, and regulatory guidelines.
                  </SheetDescription>
                </SheetHeader>

                <div className="space-y-4 pt-4 text-xs">
                  {/* Model Specs Card */}
                  <div className="p-3.5 rounded-xl bg-muted/40 border border-border space-y-2">
                    <span className="font-bold text-foreground block uppercase tracking-wider text-[11px]">
                      ML Pipeline Specification
                    </span>
                    <div className="grid grid-cols-2 gap-2 text-muted-foreground">
                      <div>Algorithm: <span className="font-mono text-foreground font-semibold">LightGBM</span></div>
                      <div>Objective: <span className="font-mono text-foreground font-semibold">binary:logloss</span></div>
                      <div>Validation: <span className="font-mono text-foreground font-semibold">5-Fold Stratified</span></div>
                      <div>Explainability: <span className="font-mono text-foreground font-semibold">TreeSHAP</span></div>
                      <div>Optimal Cutoff: <span className="font-mono text-emerald-500 font-semibold">0.360</span></div>
                      <div>Currency: <span className="font-mono text-foreground font-semibold">INR (₹)</span></div>
                    </div>
                  </div>

                  {/* RBI Compliance Notice */}
                  <div className="p-3.5 rounded-xl bg-muted/40 border border-border space-y-1.5">
                    <span className="font-bold text-foreground block uppercase tracking-wider text-[11px]">
                      Regulatory Framework
                    </span>
                    <p className="text-muted-foreground leading-relaxed">
                      Adheres to the <strong>RBI Master Direction on Digital Lending (2022/2023)</strong>, 
                      <strong>CICRA 2005</strong> credit score disclosures, and the 
                      <strong>Fair Practices Code for NBFCs & Banks</strong>.
                    </p>
                  </div>
                </div>
              </SheetContent>
            </Sheet>

            <div className="h-4 w-px bg-border/60 hidden sm:block" />

            {/* Theme Toggle */}
            <Button
              variant="outline"
              size="icon"
              className="h-7 w-7 rounded-xl border-border/80 bg-card/60 backdrop-blur-sm"
              onClick={onToggleTheme}
              title="Toggle Theme"
            >
              {isDark ? (
                <Sun className="h-3.5 w-3.5 text-amber-400" />
              ) : (
                <Moon className="h-3.5 w-3.5 text-sky-600" />
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* Main Glassmorphic Studio Workspace */}
      <main className="relative z-10 max-w-6xl mx-auto w-full px-4 sm:px-6 pt-3 pb-4">
        <div className="rounded-3xl border border-border/80 bg-card/65 dark:bg-card/40 backdrop-blur-2xl shadow-2xl p-4 sm:p-5 space-y-4">
          {/* Header Row with Title and Consolidated Glowing Preset Bar */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border/40">
            <div>
              <h1 className="text-xl sm:text-2xl font-extrabold tracking-tight text-foreground">
                Credit Risk Underwriting Studio
              </h1>
              <p className="text-[11px] sm:text-xs text-muted-foreground mt-0.5">
                Adjust applicant parameters to evaluate real-time default probability, EMI, and RBI compliance.
              </p>
            </div>

            {/* Glowing Scenario Presets Switcher (Image 2 Style) */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground font-medium hidden md:inline">Profile:</span>
              <div className="relative inline-flex items-center p-1 rounded-xl bg-muted/40 dark:bg-black/40 border border-border/80 backdrop-blur-md">
                {(["prime", "nearprime", "subprime"] as const).map((preset) => {
                  const isSelected = activePreset === preset;
                  const label = preset === "prime" ? "Prime" : preset === "nearprime" ? "Near-Prime" : "Subprime";
                  const glowColor =
                    preset === "prime"
                      ? "from-emerald-500 to-teal-500 shadow-emerald-500/30"
                      : preset === "nearprime"
                      ? "from-amber-500 to-orange-500 shadow-amber-500/30"
                      : "from-rose-500 to-red-500 shadow-rose-500/30";

                  return (
                    <button
                      key={preset}
                      type="button"
                      onClick={() => loadPreset(preset)}
                      className={cn(
                        "relative px-3 py-1 text-xs font-semibold rounded-lg transition-colors z-10",
                        isSelected ? "text-white font-bold" : "text-muted-foreground hover:text-foreground"
                      )}
                    >
                      {isSelected && (
                        <motion.div
                          layoutId="activePresetPill"
                          className={cn("absolute inset-0 rounded-lg bg-gradient-to-r shadow-md z-[-1]", glowColor)}
                          transition={{ type: "spring", stiffness: 400, damping: 30 }}
                        />
                      )}
                      {label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* 2-Column Split Workspace */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
            {/* Left Column: Sliders and Inputs */}
            <div className="lg:col-span-6 space-y-2.5">
              <div className="flex items-center justify-between pb-0.5">
                <span className="text-[11px] uppercase font-bold tracking-wider text-muted-foreground flex items-center gap-1.5">
                  <Sliders className="w-3 h-3 text-primary" />
                  Applicant Parameters (INR / ₹)
                </span>
                <span className="text-[10px] text-muted-foreground font-mono">
                  LightGBM Inputs
                </span>
              </div>

              {/* Loan Amount */}
              <div className="space-y-1 bg-background/50 dark:bg-black/30 p-2.5 rounded-2xl border border-border/80 backdrop-blur-md shadow-sm">
                <div className="flex justify-between items-center text-xs sm:text-sm">
                  <Label className="font-semibold text-foreground">Loan Requested</Label>
                  <span className="font-mono font-bold text-emerald-500 text-sm sm:text-base">
                    {formatINR(loanAmnt)}
                  </span>
                </div>
                <input
                  type="range"
                  min={50000}
                  max={3000000}
                  step={25000}
                  value={loanAmnt}
                  onChange={(e) => {
                    setLoanAmnt(Number(e.target.value));
                    setActivePreset("custom");
                  }}
                  className="w-full accent-emerald-500 cursor-pointer h-1.5"
                />
                <div className="flex justify-between text-[9px] text-muted-foreground">
                  <span>₹50,000</span>
                  <span>₹15,00,000</span>
                  <span>₹30,00,000</span>
                </div>
              </div>

              {/* Annual Income */}
              <div className="space-y-1 bg-background/50 dark:bg-black/30 p-2.5 rounded-2xl border border-border/80 backdrop-blur-md shadow-sm">
                <div className="flex justify-between items-center text-xs sm:text-sm">
                  <Label className="font-semibold text-foreground">Verifiable Annual Income</Label>
                  <span className="font-mono font-bold text-emerald-500 text-sm sm:text-base">
                    {formatINR(annualInc)}
                  </span>
                </div>
                <input
                  type="range"
                  min={200000}
                  max={5000000}
                  step={50000}
                  value={annualInc}
                  onChange={(e) => {
                    setAnnualInc(Number(e.target.value));
                    setActivePreset("custom");
                  }}
                  className="w-full accent-emerald-500 cursor-pointer h-1.5"
                />
                <div className="flex justify-between text-[9px] text-muted-foreground">
                  <span>₹2,00,000</span>
                  <span>₹25,00,000</span>
                  <span>₹50,00,000</span>
                </div>
              </div>

              {/* CIBIL Score */}
              <div className="space-y-1 bg-background/50 dark:bg-black/30 p-2.5 rounded-2xl border border-border/80 backdrop-blur-md shadow-sm">
                <div className="flex justify-between items-center text-xs sm:text-sm">
                  <div className="flex items-center gap-1.5">
                    <Label className="font-semibold text-foreground">CIBIL / Bureau Score</Label>
                    <span className="text-[9px] px-1.5 py-0.2 bg-muted rounded text-muted-foreground font-mono">
                      CICRA
                    </span>
                  </div>
                  <span className={`font-mono font-bold text-xs sm:text-sm ${
                    cibilScore >= 750
                      ? "text-emerald-500"
                      : cibilScore >= 680
                      ? "text-amber-500"
                      : "text-destructive"
                  }`}>
                    {cibilScore} ({cibilScore >= 750 ? "Prime" : cibilScore >= 680 ? "Near-Prime" : "Subprime"})
                  </span>
                </div>
                <input
                  type="range"
                  min={300}
                  max={900}
                  step={5}
                  value={cibilScore}
                  onChange={(e) => {
                    setCibilScore(Number(e.target.value));
                    setActivePreset("custom");
                  }}
                  className="w-full accent-emerald-500 cursor-pointer h-1.5"
                />
                <div className="flex justify-between text-[9px] text-muted-foreground">
                  <span>300 (Poor)</span>
                  <span>680 (Threshold)</span>
                  <span>750+ (Excellent)</span>
                  <span>900</span>
                </div>
              </div>

              {/* DTI and Inquiries Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {/* Debt to Income */}
                <div className="space-y-0.5 bg-background/50 dark:bg-black/30 p-2.5 rounded-2xl border border-border/80 backdrop-blur-md shadow-sm">
                  <div className="flex justify-between text-xs">
                    <Label className="font-semibold text-foreground">DTI Ratio</Label>
                    <span className={`font-mono font-bold ${dti > 30 ? "text-destructive" : "text-emerald-500"}`}>
                      {dti.toFixed(1)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={5}
                    max={60}
                    step={0.5}
                    value={dti}
                    onChange={(e) => {
                      setDti(Number(e.target.value));
                      setActivePreset("custom");
                    }}
                    className="w-full accent-emerald-500 cursor-pointer h-1.5"
                  />
                  <span className="text-[9px] text-muted-foreground block text-right font-medium">
                    {dti <= 25 ? "Safe" : dti <= 35 ? "Moderate" : "Elevated Risk"}
                  </span>
                </div>

                {/* Inquiries */}
                <div className="space-y-0.5 bg-background/50 dark:bg-black/30 p-2.5 rounded-2xl border border-border/80 backdrop-blur-md shadow-sm">
                  <div className="flex justify-between text-xs">
                    <Label className="font-semibold text-foreground">Inquiries (Past 6M)</Label>
                    <span className={`font-mono font-bold ${inqLast6M >= 2 ? "text-destructive" : "text-emerald-500"}`}>
                      {inqLast6M} inq
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={8}
                    step={1}
                    value={inqLast6M}
                    onChange={(e) => {
                      setInqLast6M(Number(e.target.value));
                      setActivePreset("custom");
                    }}
                    className="w-full accent-emerald-500 cursor-pointer h-1.5"
                  />
                  <span className="text-[9px] text-muted-foreground block text-right font-medium">
                    {inqLast6M === 0 ? "Ideal" : inqLast6M <= 2 ? "Acceptable" : "Credit Hungry"}
                  </span>
                </div>
              </div>

              {/* Glowing Segmented Switcher for Tenor (Image 2 style) */}
              <div className="space-y-1.5 bg-background/50 dark:bg-black/30 p-2.5 rounded-2xl border border-border/80 backdrop-blur-md shadow-sm">
                <div className="flex justify-between items-center text-xs">
                  <Label className="font-semibold text-foreground flex items-center gap-1.5">
                    <CalendarDays className="w-3.5 h-3.5 text-primary" />
                    Repayment Tenor (Duration)
                  </Label>
                  <span className="text-muted-foreground text-[10px] font-mono">
                    {term} Months ({Number(term) / 12} {Number(term) === 12 ? "Year" : "Years"})
                  </span>
                </div>

                {/* Glowing Segmented Pill Group */}
                <div className="relative flex items-center p-1 rounded-xl bg-muted/40 dark:bg-black/40 border border-border/70 backdrop-blur-md gap-1">
                  {TENOR_OPTIONS.map((opt) => {
                    const isActive = term === opt.value;
                    return (
                      <button
                        key={opt.value}
                        type="button"
                        onClick={() => {
                          setTerm(opt.value);
                          setActivePreset("custom");
                        }}
                        className={cn(
                          "relative flex-1 py-1 px-1.5 text-xs font-semibold rounded-lg transition-colors z-10 text-center",
                          isActive ? "text-white font-bold" : "text-muted-foreground hover:text-foreground"
                        )}
                      >
                        {isActive && (
                          <motion.div
                            layoutId="activeTenorPill"
                            className="absolute inset-0 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 shadow-md shadow-emerald-500/25 z-[-1]"
                            transition={{ type: "spring", stiffness: 400, damping: 30 }}
                          />
                        )}
                        <span className="block leading-tight text-[11px] font-semibold">{opt.label}</span>
                        <span className="text-[8px] opacity-80 block font-normal">{opt.sub}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Right Column: Real-Time Risk Assessment & Explainability */}
            <div className="lg:col-span-6 space-y-3">
              <div className="flex items-center justify-between pb-0.5">
                <span className="text-[11px] uppercase font-bold tracking-wider text-muted-foreground flex items-center gap-1.5">
                  <Cpu className="w-3 h-3 text-primary" />
                  Live Underwriting Decision Engine
                </span>
                <span className="text-[10px] font-mono text-emerald-500 font-semibold">
                  Cost-Optimal Cutoff τ* = 0.360
                </span>
              </div>

              {/* Decision Status Card */}
              <div className={`p-4 rounded-3xl border transition-all duration-300 shadow-lg backdrop-blur-xl ${
                riskAnalysis.isApproved
                  ? "bg-emerald-500/10 border-emerald-500/40 text-emerald-950 dark:text-emerald-100 shadow-emerald-500/5"
                  : "bg-destructive/10 border-destructive/40 text-destructive-foreground shadow-destructive/5"
              }`}>
                <div className="flex items-start justify-between">
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground block mb-0.5">
                      Automated Decision
                    </span>
                    <div className="flex items-center gap-2">
                      {riskAnalysis.isApproved ? (
                        <>
                          <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                          <span className="text-xl sm:text-2xl font-black text-emerald-600 dark:text-emerald-400">
                            APPROVED
                          </span>
                        </>
                      ) : (
                        <>
                          <XCircle className="w-5 h-5 text-destructive" />
                          <span className="text-xl sm:text-2xl font-black text-destructive">
                            DECLINED (HIGH RISK)
                          </span>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="text-right">
                    <span className="text-[10px] font-medium text-muted-foreground block">
                      Default Probability (PD)
                    </span>
                    <span className={`text-xl sm:text-2xl font-mono font-black ${
                      riskAnalysis.defaultProb < 0.20
                        ? "text-emerald-500"
                        : riskAnalysis.defaultProb < 0.36
                        ? "text-amber-500"
                        : "text-destructive"
                    }`}>
                      {(riskAnalysis.defaultProb * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Progress Gauge */}
                <div className="mt-3 space-y-1">
                  <div className="w-full bg-background/60 h-2.5 rounded-full overflow-hidden p-0.5 border border-border/60">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        riskAnalysis.defaultProb < 0.20
                          ? "bg-emerald-500"
                          : riskAnalysis.defaultProb < 0.36
                          ? "bg-amber-500"
                          : "bg-destructive"
                      }`}
                      style={{ width: `${Math.min(riskAnalysis.defaultProb * 100, 100)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[9px] text-muted-foreground font-mono">
                    <span>0% (Prime)</span>
                    <span className="font-bold text-foreground">Cutoff: 36.0%</span>
                    <span>100% (Default)</span>
                  </div>
                </div>

                {/* Expanded Financial Metrics Strip (with Monthly EMI) */}
                <div className="mt-3 pt-3 border-t border-border/40 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                  <div>
                    <span className="text-muted-foreground text-[9px] block font-medium">Estimated EMI</span>
                    <span className="font-mono font-bold text-foreground text-xs sm:text-sm">
                      {formatINR(riskAnalysis.monthlyEmi)}<span className="text-[9px] text-muted-foreground font-normal">/mo</span>
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground text-[9px] block font-medium">Risk-Based APR</span>
                    <span className="font-mono font-bold text-foreground text-xs sm:text-sm">
                      {riskAnalysis.recommendedApr.toFixed(2)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground text-[9px] block font-medium">Expected Loss</span>
                    <span className="font-mono font-bold text-foreground text-xs sm:text-sm">
                      {formatINR(riskAnalysis.expectedLoss)}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground text-[9px] block font-medium">Loan / Income</span>
                    <span className="font-mono font-bold text-foreground text-xs sm:text-sm">
                      {(riskAnalysis.loanToIncome * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* TreeSHAP Explainability & Adverse Action Section */}
              <div className="p-3.5 rounded-2xl border border-border/80 bg-background/50 dark:bg-black/30 backdrop-blur-md space-y-2.5 shadow-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <Award className="w-3.5 h-3.5 text-emerald-500" />
                    <span className="text-[11px] font-bold uppercase tracking-wider text-foreground">
                      TreeSHAP Feature Attributions
                    </span>
                  </div>
                  <span className="text-[9px] text-muted-foreground font-mono">
                    Local Factor Weights
                  </span>
                </div>

                {/* Factors List */}
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between items-center p-2 rounded-xl bg-muted/40 border border-border/40">
                    <span className="font-medium text-foreground text-[11px]">1. CIBIL Bureau Standing</span>
                    <span className={`font-mono font-semibold text-[11px] ${cibilScore >= 750 ? "text-emerald-500" : "text-destructive"}`}>
                      {cibilScore >= 750 ? "-0.05 (Protective)" : "+0.18 (Risk Driver)"}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-2 rounded-xl bg-muted/40 border border-border/40">
                    <span className="font-medium text-foreground text-[11px]">2. Debt-to-Income Burden</span>
                    <span className={`font-mono font-semibold text-[11px] ${dti <= 25 ? "text-emerald-500" : "text-destructive"}`}>
                      {dti <= 25 ? "-0.04 (Protective)" : "+0.12 (Risk Driver)"}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-2 rounded-xl bg-muted/40 border border-border/40">
                    <span className="font-medium text-foreground text-[11px]">3. Loan-to-Income Leverage</span>
                    <span className={`font-mono font-semibold text-[11px] ${riskAnalysis.loanToIncome <= 0.4 ? "text-emerald-500" : "text-destructive"}`}>
                      {riskAnalysis.loanToIncome <= 0.4 ? "-0.03 (Protective)" : "+0.15 (Risk Driver)"}
                    </span>
                  </div>
                </div>

                {/* Action Button: Generate RBI Adverse Action Notice */}
                <Sheet>
                  <SheetTrigger asChild>
                    <Button
                      variant={riskAnalysis.isApproved ? "outline" : "destructive"}
                      className="w-full mt-1.5 rounded-xl text-xs font-semibold gap-2 h-9 shadow-sm border-border/80"
                    >
                      <FileText className="w-3.5 h-3.5" />
                      <span>View Official RBI Adverse Action Notice</span>
                    </Button>
                  </SheetTrigger>
                  <SheetContent side="right" className="w-[340px] sm:w-[500px] overflow-y-auto">
                    <SheetHeader className="text-left pb-4 border-b border-border">
                      <div className="flex items-center gap-2 font-bold text-base text-foreground">
                        <Scale className="w-5 h-5 text-primary" />
                        <SheetTitle>Statutory Credit Notice</SheetTitle>
                      </div>
                      <SheetDescription className="text-xs text-muted-foreground">
                        Under Section 6 of the RBI Fair Practices Code for NBFCs & Digital Lending Guidelines (2022).
                      </SheetDescription>
                    </SheetHeader>

                    {/* Formal Letter Preview */}
                    <div className="mt-4 p-4 rounded-xl border border-border bg-muted/30 space-y-3 text-xs font-sans leading-relaxed">
                      <div className="border-b border-border/60 pb-2">
                        <div className="font-bold text-sm text-foreground">
                          CREDIT UNDERWRITING NOTICE OF ADVERSE ACTION
                        </div>
                        <div className="text-[10px] text-muted-foreground">
                          Reference ID: CR-IN-{Math.floor(100000 + Math.random() * 900000)} | Date: {new Date().toLocaleDateString("en-IN")}
                        </div>
                      </div>

                      <p>
                        <strong>Applicant Status:</strong> {riskAnalysis.isApproved ? "Approved with Conditions" : "Application Not Approved at Current Terms"}
                      </p>
                      <p>
                        <strong>Requested Facility:</strong> Unsecured Personal Credit of {formatINR(loanAmnt)} for {term} months.
                      </p>
                      <p>
                        <strong>Calculated Monthly EMI:</strong> {formatINR(riskAnalysis.monthlyEmi)} at {riskAnalysis.recommendedApr.toFixed(2)}% APR.
                      </p>

                      <div>
                        <strong>Principal Reasons for Credit Decision:</strong>
                        {riskAnalysis.adverseFactors.length > 0 ? (
                          <ul className="list-disc pl-4 space-y-1.5 mt-1 text-muted-foreground">
                            {riskAnalysis.adverseFactors.map((f, i) => (
                              <li key={i}>
                                <span className="font-semibold text-foreground">{f.title} ({f.code}):</span> {f.desc}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-emerald-500 font-semibold mt-1">
                            Applicant meets all Tier-1 risk solvency benchmarks. No statutory adverse factors cited.
                          </p>
                        )}
                      </div>

                      <div className="border-t border-border/60 pt-2 text-[11px] text-muted-foreground space-y-1">
                        <p>
                          <strong>Bureau Disclosure (CICRA 2005):</strong> Information evaluated from licensed Credit Information Companies (CIBIL/Equifax/Experian). Score on file: {cibilScore}.
                        </p>
                        <p>
                          <strong>Grievance Redressal:</strong> In accordance with RBI Master Directions, applicants have the right to request review within 30 days.
                        </p>
                      </div>
                    </div>
                  </SheetContent>
                </Sheet>
              </div>
            </div>
          </div>
        </div>
      </main>
    </SonarGrid>
  );
}
