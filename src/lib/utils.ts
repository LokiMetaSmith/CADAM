import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { ModelConfig } from '@/types/misc';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const PARAMETRIC_MODELS: ModelConfig[] = [
  {
    id: 'anthropic-fast',
    name: 'Adam',
    description: 'Fast responses, optimized for iterative part design',
  },
  {
    id: 'anthropic-quality',
    name: 'Adam Pro',
    description: 'Enhanced capabilities takes longer to think',
  },
  
    id: 'grok',
    name: 'Grok',
    description: 'Language model from xAI',
  },
  {
    id: 'google',
    name: 'Google',
    description: 'Language model from Google',
  },
  {
    id: 'llama',
    name: 'Llama',
    description: 'Language model from Meta',
  },
];
