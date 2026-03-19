import type { Meta, StoryObj } from '@storybook/react';
import RegimeHeatmap from './RegimeHeatmap';

const meta: Meta<typeof RegimeHeatmap> = {
  title: 'Components/RegimeHeatmap',
  component: RegimeHeatmap,
  parameters: {
    layout: 'fullscreen',
  },
  argTypes: {
    beta: {
      control: { type: 'range', min: 0, max: 0.9999, step: 0.0001 },
      description: 'Market velocity relative to speed of light (β)',
    },
  },
};

export default meta;
type Story = StoryObj<typeof RegimeHeatmap>;

export const Default: Story = {
  args: {
    beta: 0.5,
  },
};

export const HighBeta: Story = {
  name: 'High Beta (0.9) — mostly SPACELIKE',
  args: {
    beta: 0.9,
  },
};

export const LowBeta: Story = {
  name: 'Low Beta (0.1) — mostly TIMELIKE',
  args: {
    beta: 0.1,
  },
};
