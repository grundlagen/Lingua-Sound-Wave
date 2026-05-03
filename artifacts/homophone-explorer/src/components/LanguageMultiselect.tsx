import { useState } from "react";
import { Check, ChevronsUpDown, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { cn } from "@/lib/utils";
import type { Language } from "@workspace/api-client-react";

interface Props {
  languages: Language[];
  selected: string[];
  onChange: (next: string[]) => void;
  exclude?: string;
  placeholder?: string;
}

export function LanguageMultiselect({ languages, selected, onChange, exclude, placeholder = "All languages" }: Props) {
  const [open, setOpen] = useState(false);
  const visible = languages.filter((l) => l.code !== exclude);

  const toggle = (code: string) => {
    if (selected.includes(code)) onChange(selected.filter((c) => c !== code));
    else onChange([...selected, code]);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between min-h-10 h-auto py-2"
          data-testid="lang-multiselect"
        >
          <div className="flex flex-wrap gap-1 items-center">
            {selected.length === 0 ? (
              <span className="text-muted-foreground">{placeholder}</span>
            ) : (
              selected.slice(0, 4).map((code) => {
                const l = languages.find((x) => x.code === code);
                return (
                  <Badge key={code} variant="secondary" className="text-xs">
                    {l?.name ?? code}
                    <X
                      className="ml-1 h-3 w-3 cursor-pointer"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggle(code);
                      }}
                    />
                  </Badge>
                );
              })
            )}
            {selected.length > 4 ? (
              <Badge variant="secondary" className="text-xs">
                +{selected.length - 4} more
              </Badge>
            ) : null}
          </div>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[320px] p-0" align="start">
        <Command>
          <CommandInput placeholder="Search languages..." />
          <CommandList>
            <CommandEmpty>No languages.</CommandEmpty>
            <CommandGroup>
              {visible.map((l) => (
                <CommandItem
                  key={l.code}
                  value={`${l.name} ${l.nativeName} ${l.code}`}
                  onSelect={() => toggle(l.code)}
                  data-testid={`lang-option-${l.code}`}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      selected.includes(l.code) ? "opacity-100" : "opacity-0",
                    )}
                  />
                  <span className="flex-1">{l.name}</span>
                  <span className="text-xs text-muted-foreground">{l.nativeName}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
