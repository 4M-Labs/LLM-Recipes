# Monitoring Pattern for LLM Applications

A comprehensive guide to implementing effective monitoring, observability, and evaluation systems for LLM applications.

## Overview

The Monitoring Pattern provides a structured approach to tracking performance, quality, and costs in LLM applications. It enables real-time insights, quality assurance, and continuous improvement.

## Core Components

### 1. Performance Monitoring
```typescript
interface PerformanceMetrics {
  latency: number;
  tokensUsed: number;
  requestSize: number;
  responseSize: number;
  modelName: string;
  timestamp: number;
}

class PerformanceMonitor {
  private metrics: PerformanceMetrics[] = [];
  
  recordMetrics(metrics: PerformanceMetrics): void {
    this.metrics.push({
      ...metrics,
      timestamp: Date.now()
    });
  }
  
  getAverageLatency(timeWindowMs: number): number {
    const now = Date.now();
    const relevantMetrics = this.metrics.filter(
      m => now - m.timestamp <= timeWindowMs
    );
    
    if (relevantMetrics.length === 0) return 0;
    
    return relevantMetrics.reduce((sum, m) => sum + m.latency, 0) / relevantMetrics.length;
  }
  
  getTokenUsage(timeWindowMs: number): number {
    const now = Date.now();
    return this.metrics
      .filter(m => now - m.timestamp <= timeWindowMs)
      .reduce((sum, m) => sum + m.tokensUsed, 0);
  }
}
```

### 2. Quality Assurance
```typescript
interface QualityMetrics {
  responseId: string;
  prompt: string;
  response: string;
  expectedBehaviors: string[];
  satisfiedBehaviors: string[];
  score: number;
  feedback?: string;
}

class QualityMonitor {
  private evaluations: QualityMetrics[] = [];
  
  async evaluateResponse(metrics: QualityMetrics): Promise<void> {
    // Evaluate response against expected behaviors
    const score = await this.calculateQualityScore(metrics);
    
    this.evaluations.push({
      ...metrics,
      score
    });
  }
  
  private async calculateQualityScore(metrics: QualityMetrics): Promise<number> {
    const totalBehaviors = metrics.expectedBehaviors.length;
    const satisfiedCount = metrics.satisfiedBehaviors.length;
    
    return satisfiedCount / totalBehaviors;
  }
  
  getAverageQualityScore(timeWindowMs: number): number {
    const now = Date.now();
    const relevantEvaluations = this.evaluations.filter(
      e => now - e.timestamp <= timeWindowMs
    );
    
    if (relevantEvaluations.length === 0) return 0;
    
    return relevantEvaluations.reduce((sum, e) => sum + e.score, 0) / 
      relevantEvaluations.length;
  }
}
```

### 3. Cost Tracking
```typescript
interface CostMetrics {
  modelName: string;
  tokensUsed: number;
  costPerToken: number;
  timestamp: number;
}

class CostMonitor {
  private costs: CostMetrics[] = [];
  
  recordCost(metrics: CostMetrics): void {
    this.costs.push({
      ...metrics,
      timestamp: Date.now()
    });
  }
  
  getTotalCost(timeWindowMs: number): number {
    const now = Date.now();
    return this.costs
      .filter(c => now - c.timestamp <= timeWindowMs)
      .reduce((sum, c) => sum + (c.tokensUsed * c.costPerToken), 0);
  }
  
  getCostByModel(modelName: string, timeWindowMs: number): number {
    const now = Date.now();
    return this.costs
      .filter(c => c.modelName === modelName && now - c.timestamp <= timeWindowMs)
      .reduce((sum, c) => sum + (c.tokensUsed * c.costPerToken), 0);
  }
}
```

### 4. Error Tracking
```typescript
interface ErrorMetrics {
  errorType: string;
  message: string;
  component: string;
  stackTrace?: string;
  context?: any;
  timestamp: number;
}

class ErrorMonitor {
  private errors: ErrorMetrics[] = [];
  
  recordError(error: ErrorMetrics): void {
    this.errors.push({
      ...error,
      timestamp: Date.now()
    });
  }
  
  getErrorRate(timeWindowMs: number): number {
    const now = Date.now();
    const recentErrors = this.errors.filter(
      e => now - e.timestamp <= timeWindowMs
    ).length;
    
    return recentErrors / (timeWindowMs / 1000); // errors per second
  }
  
  getErrorsByType(errorType: string, timeWindowMs: number): ErrorMetrics[] {
    const now = Date.now();
    return this.errors.filter(
      e => e.errorType === errorType && now - e.timestamp <= timeWindowMs
    );
  }
}
```

## Integration Example

```typescript
class MonitoringSystem {
  constructor(
    private performanceMonitor: PerformanceMonitor,
    private qualityMonitor: QualityMonitor,
    private costMonitor: CostMonitor,
    private errorMonitor: ErrorMonitor,
    private alertThresholds: AlertThresholds
  ) {}
  
  async monitorRequest(
    request: LLMRequest,
    response: LLMResponse,
    context: RequestContext
  ): Promise<void> {
    // Record performance metrics
    this.performanceMonitor.recordMetrics({
      latency: context.endTime - context.startTime,
      tokensUsed: response.usage.totalTokens,
      requestSize: request.prompt.length,
      responseSize: response.text.length,
      modelName: request.model,
      timestamp: Date.now()
    });
    
    // Evaluate quality
    await this.qualityMonitor.evaluateResponse({
      responseId: response.id,
      prompt: request.prompt,
      response: response.text,
      expectedBehaviors: request.expectedBehaviors,
      satisfiedBehaviors: await this.checkBehaviors(response.text, request.expectedBehaviors),
      score: 0 // Will be calculated by evaluateResponse
    });
    
    // Track costs
    this.costMonitor.recordCost({
      modelName: request.model,
      tokensUsed: response.usage.totalTokens,
      costPerToken: this.getModelCost(request.model),
      timestamp: Date.now()
    });
    
    // Check for alerts
    await this.checkAlertThresholds();
  }
  
  private async checkAlertThresholds(): Promise<void> {
    const timeWindow = 5 * 60 * 1000; // 5 minutes
    
    // Check performance
    const avgLatency = this.performanceMonitor.getAverageLatency(timeWindow);
    if (avgLatency > this.alertThresholds.maxLatency) {
      await this.sendAlert('High Latency', `Average latency: ${avgLatency}ms`);
    }
    
    // Check quality
    const qualityScore = this.qualityMonitor.getAverageQualityScore(timeWindow);
    if (qualityScore < this.alertThresholds.minQualityScore) {
      await this.sendAlert('Low Quality', `Quality score: ${qualityScore}`);
    }
    
    // Check costs
    const costRate = this.costMonitor.getTotalCost(timeWindow);
    if (costRate > this.alertThresholds.maxCostRate) {
      await this.sendAlert('High Cost', `Cost rate: $${costRate}/5min`);
    }
    
    // Check errors
    const errorRate = this.errorMonitor.getErrorRate(timeWindow);
    if (errorRate > this.alertThresholds.maxErrorRate) {
      await this.sendAlert('High Error Rate', `Error rate: ${errorRate}/sec`);
    }
  }
}
```

## Best Practices

### 1. Metrics Collection
- Collect comprehensive metrics across all system components
- Use structured logging for easy analysis
- Implement proper sampling for high-volume metrics
- Store metrics with appropriate retention policies

### 2. Alerting Strategy
```typescript
interface AlertConfig {
  name: string;
  condition: (metrics: any) => boolean;
  message: (metrics: any) => string;
  severity: 'low' | 'medium' | 'high';
  cooldown: number;
}

class AlertManager {
  private lastAlerts: Map<string, number> = new Map();
  
  async checkAlert(config: AlertConfig, metrics: any): Promise<void> {
    const now = Date.now();
    const lastAlert = this.lastAlerts.get(config.name) || 0;
    
    if (now - lastAlert < config.cooldown) {
      return;
    }
    
    if (config.condition(metrics)) {
      await this.sendAlert({
        name: config.name,
        message: config.message(metrics),
        severity: config.severity,
        timestamp: now
      });
      
      this.lastAlerts.set(config.name, now);
    }
  }
}
```

### 3. Visualization
```typescript
interface TimeSeriesData {
  timestamp: number;
  value: number;
}

class MetricsVisualizer {
  generateTimeSeries(
    metrics: TimeSeriesData[],
    timeWindowMs: number
  ): TimeSeriesData[] {
    const now = Date.now();
    return metrics
      .filter(m => now - m.timestamp <= timeWindowMs)
      .sort((a, b) => a.timestamp - b.timestamp);
  }
  
  calculateMovingAverage(
    series: TimeSeriesData[],
    windowSize: number
  ): TimeSeriesData[] {
    const result: TimeSeriesData[] = [];
    
    for (let i = windowSize - 1; i < series.length; i++) {
      const windowSum = series
        .slice(i - windowSize + 1, i + 1)
        .reduce((sum, point) => sum + point.value, 0);
      
      result.push({
        timestamp: series[i].timestamp,
        value: windowSum / windowSize
      });
    }
    
    return result;
  }
}
```

## Performance Considerations

### 1. Efficient Storage
```typescript
class MetricsStore {
  private readonly maxSize: number;
  private readonly pruneThreshold: number;
  
  constructor(maxSize: number, pruneThreshold: number) {
    this.maxSize = maxSize;
    this.pruneThreshold = pruneThreshold;
  }
  
  addMetrics(metrics: any[]): void {
    // Add new metrics
    this.metrics.push(...metrics);
    
    // Check if pruning is needed
    if (this.metrics.length > this.maxSize) {
      this.pruneOldMetrics();
    }
  }
  
  private pruneOldMetrics(): void {
    const now = Date.now();
    this.metrics = this.metrics.filter(
      m => now - m.timestamp <= this.pruneThreshold
    );
  }
}
```

### 2. Sampling Strategy
```typescript
class MetricsSampler {
  private samplingRate: number;
  
  constructor(samplingRate: number) {
    this.samplingRate = samplingRate;
  }
  
  shouldSample(): boolean {
    return Math.random() < this.samplingRate;
  }
  
  adjustSamplingRate(currentLoad: number, targetLoad: number): void {
    this.samplingRate = Math.min(1, this.samplingRate * (targetLoad / currentLoad));
  }
}
```

## Example Dashboard Implementation

```typescript
class MonitoringDashboard {
  constructor(
    private performanceMonitor: PerformanceMonitor,
    private qualityMonitor: QualityMonitor,
    private costMonitor: CostMonitor,
    private errorMonitor: ErrorMonitor,
    private visualizer: MetricsVisualizer
  ) {}
  
  async generateDashboard(timeWindowMs: number): Promise<Dashboard> {
    return {
      performance: {
        averageLatency: this.performanceMonitor.getAverageLatency(timeWindowMs),
        latencyTrend: this.visualizer.generateTimeSeries(
          this.performanceMonitor.getLatencyHistory(),
          timeWindowMs
        ),
        tokenUsage: this.performanceMonitor.getTokenUsage(timeWindowMs)
      },
      quality: {
        averageScore: this.qualityMonitor.getAverageQualityScore(timeWindowMs),
        scoreTrend: this.visualizer.generateTimeSeries(
          this.qualityMonitor.getScoreHistory(),
          timeWindowMs
        )
      },
      costs: {
        totalCost: this.costMonitor.getTotalCost(timeWindowMs),
        costByModel: await this.getCostBreakdown(timeWindowMs)
      },
      errors: {
        errorRate: this.errorMonitor.getErrorRate(timeWindowMs),
        errorsByType: await this.getErrorBreakdown(timeWindowMs)
      }
    };
  }
}
```

## Conclusion

The Monitoring Pattern is essential for maintaining healthy LLM applications. Key takeaways:

- Implement comprehensive metrics collection
- Monitor performance, quality, costs, and errors
- Set up proper alerting with appropriate thresholds
- Visualize metrics for easy analysis
- Use efficient storage and sampling strategies
- Regularly review and adjust monitoring parameters

## Resources

- [Google SRE Book](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
