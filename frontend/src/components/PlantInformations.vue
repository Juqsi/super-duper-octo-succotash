<script lang="ts" setup>
import { ref } from 'vue'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Droplet, GaugeCircle, Leaf, Ruler, Sparkles, Sun, TreePine } from 'lucide-vue-next'
import { useRoute } from 'vue-router'

const route = useRoute()
const plantId = route.params.id

interface Plant {
  plant_id: number
  common_name: string | undefined
  scientific_name: string | undefined
  cycle: string | undefined
  watering_short: string | undefined
  hardiness_zone: string | undefined
  sun: string | undefined
  cones: string | undefined
  leaf: string | undefined
  leaf_color: string | undefined
  growth_rate: string | undefined
  care_level: string | undefined
  watering_extended: string | undefined
  sunlight_extended: string | undefined
  pruning_extended: string | undefined
}

const plant: Plant = ref({
  care_level: null,
  common_name: 'windflower',
  cones: null,
  cycle: 'Herbaceous Perennial',
  description:
    'Info, guide, description - Windflower (Anemone Narcissiflora) with images, care guide, remedies, question & related plants',
  growth_rate: 'Low',
  hardiness_zone: '4 - 7',
  id: 826,
  leaf: 'Yes',
  leaf_color: 'green',
  plant_id: 825,
  pruning_extended:
    'Windflower should be pruned in late winter to early spring. Pruning should be limited to removing the dead flower stalks and no more than 1 third of the foliage. To prevent plants from becoming leggy, lightly prune the plant at a diameter of 30cm. Thin out any tall, weak or dead growth. It is not recommended to prune dead miniature cultivars.',
  scientific_name: 'Anemone Narcissiflora',
  sun: 'Full sun,part shade',
  sunlight_extended:
    'Windflower (Anemone narcissiflora) should be exposed to sunlight for 4-6 hours of direct sunlight per day for optimal growth and health. It should be placed in full sun, in a location with unobstructed access to the sun. Morning sun is best and afternoon shade is preferred as midday sun can cause the plants leaves to sunburn. Windflower can handle partial shade but it may become more susceptible to disease and pests, and may flower less without enough sun.',
  watering_extended:
    'In general, Anemone narcissiflora requires regular watering to keep its soil lightly moist. Water once a week or as needed in order to keep the soil slightly damp and not soggy. During periods of extended heat or drought, water may need to be applied more frequently. Windflowers typically do not need too much water, however, as it can lead to root rot in some cases. Try to water around the base of the plant, letting the water soak into the soil rather than splashing onto the foliage. Too much water on the leaves can result in fungal issues.',
  watering_short: 'Average',
})
</script>

<template>
  <Card class="max-w-2xl w-full rounded-2xl shadow-lg p-4 bg-white dark:bg-gray-950">
    <CardHeader>
      <CardTitle class="text-2xl font-bold">{{ plant.common_name }}</CardTitle>
      <CardDescription class="text-muted-foreground italic">
        {{ plant.scientific_name }}
      </CardDescription>
    </CardHeader>
    <CardContent class="space-y-6">
      <!-- Overview Info Grid -->
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div class="flex items-center gap-3">
          <GaugeCircle class="text-green-600 w-5 h-5" />
          <span><strong>Cycle:</strong> {{ plant.cycle }}</span>
        </div>
        <div class="flex items-center gap-3">
          <Droplet class="text-blue-500 w-5 h-5" />
          <span><strong>Watering:</strong> {{ plant.watering_short }}</span>
        </div>
        <div class="flex items-center gap-3">
          <Sun class="text-yellow-500 w-5 h-5" />
          <span><strong>Sun:</strong> {{ plant.sun }}</span>
        </div>
        <div class="flex items-center gap-3">
          <TreePine class="text-emerald-600 w-5 h-5" />
          <span><strong>Hardiness Zone:</strong> {{ plant.hardiness_zone }}</span>
        </div>
        <div class="flex items-center gap-3">
          <Leaf class="text-lime-600 w-5 h-5" />
          <span><strong>Leaf:</strong> {{ plant.leaf }}</span>
        </div>
        <div class="flex items-center gap-3">
          <Sparkles class="text-purple-500 w-5 h-5" />
          <span><strong>Growth Rate:</strong> {{ plant.growth_rate }}</span>
        </div>
        <div class="flex items-center gap-3">
          <Ruler class="text-gray-500 w-5 h-5" />
          <span><strong>Care Level:</strong> {{ plant.care_level }}</span>
        </div>
      </div>

      <!-- Accordion for Extended Info -->
      <Accordion class="mt-4 border-t pt-4 w-full max-w-full" collapsible>
        <AccordionItem value="watering">
          <AccordionTrigger>
            <div class="flex items-center gap-2">
              <Droplet class="w-4 h-4" />
              Watering Details
            </div>
          </AccordionTrigger>
          <AccordionContent class="text-sm text-muted-foreground">
            {{ plant.watering_extended }}
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="sunlight">
          <AccordionTrigger>
            <div class="flex items-center gap-2">
              <Sun class="w-4 h-4" />
              Sunlight Requirements
            </div>
          </AccordionTrigger>
          <AccordionContent class="text-sm text-muted-foreground">
            {{ plant.sunlight_extended }}
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="pruning">
          <AccordionTrigger>
            <div class="flex items-center gap-2">
              <Leaf class="w-4 h-4" />
              Pruning Guide
            </div>
          </AccordionTrigger>
          <AccordionContent class="text-sm text-muted-foreground">
            {{ plant.pruning_extended }}
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </CardContent>
    <CardFooter>
      <CardDescription class="text-center w-full">
        <a
          :href="
            'https://perenual.com/plant-species-database-search-finder/species/' + plant.plant_id
          "
          class="text-blue-400"
          >more information</a
        >
      </CardDescription>
    </CardFooter>
  </Card>
</template>

<style scoped></style>
