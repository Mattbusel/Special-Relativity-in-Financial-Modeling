import type { Meta, StoryObj } from '@storybook/react';
import BacktestPanel from './BacktestPanel';

const meta: Meta<typeof BacktestPanel> = {
  title: 'Components/BacktestPanel',
  component: BacktestPanel,
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
type Story = StoryObj<typeof BacktestPanel>;

export const Default: Story = {
  args: {
    beta: 0.5,
  },
};

export const HighBeta: Story = {
  name: 'High Beta (0.9)',
  args: {
    beta: 0.9,
  },
};

export const LowBeta: Story = {
  name: 'Low Beta (0.1)',
  args: {
    beta: 0.1,
  },
};
