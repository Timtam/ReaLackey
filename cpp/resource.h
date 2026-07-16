// Control / dialog IDs. Plain #defines only: consumed by rc.exe on Windows AND
// by swell_resgen.php on macOS/Linux. Keep it ASCII and free of anything but
// #define lines. IDOK / IDCANCEL come from windows.h / swell.h.
#ifndef RAAI_RESOURCE_H
#define RAAI_RESOURCE_H

#define ID_ASSISTANT_DLG   1001
#define ID_OUTPUT_EDIT     1002   // read-only, scrollable conversation log
#define ID_STATUS_TEXT     1003   // single-line status field
#define ID_INPUT_EDIT      1004   // user input
#define ID_SUBMIT_BTN      1005   // "Send"
#define ID_CONFIRM_BTN     1006   // "Confirm" (reserved for Phase 3 mutations)

// Provider management dialog (Phase 5, M4).
#define ID_PROVIDERS_DLG   1010
#define ID_PROV_LIST       1011   // listbox of configured providers
#define ID_PROV_ADD        1012
#define ID_PROV_EDIT       1013
#define ID_PROV_DELETE     1014
#define ID_PROV_DEFAULT    1015   // "Set as default"

// Provider settings dialog (Phase 5, M5): add / edit one account, with a
// "Fetch models" button next to the model field.
#define ID_PROVIDER_EDIT_DLG 1020
#define ID_PE_LABEL          1021
#define ID_PE_BASEURL        1022
#define ID_PE_BASEURL_LBL    1023   // "Base URL:" label (hidden for Anthropic)
#define ID_PE_MODEL          1024
#define ID_PE_FETCH          1025   // "Fetch models..." button
#define ID_PE_MAXTOK         1026
#define ID_PE_VISION         1027   // "Supports images" checkbox
#define ID_PE_KEY            1028   // "Add key" input (typed key -> Add button)
#define ID_PE_KEYHINT        1029   // caption under the key list
#define ID_PE_MAXTURNS       1030   // "Tool steps:" (max agentic turns) field
// Multi-key list: an ordered list of API keys (top tried first, falls back to
// the next on a per-key limit). Managed with Add / Delete / Move up / Move down.
#define ID_PE_KEYLIST        1031   // listbox of configured keys (masked)
#define ID_PE_KEYADD         1032   // add the typed key to the list
#define ID_PE_KEYDEL         1033   // remove the selected key
#define ID_PE_KEYUP          1034   // move the selected key up (higher priority)
#define ID_PE_KEYDOWN        1035   // move the selected key down
#define ID_PE_AUDIO          1036   // "Supports audio (listening)" checkbox
#define ID_PE_THINKING       1037   // "Extended thinking (reasoning)" checkbox (Anthropic)

// Prompt presets: reusable prompt bodies inserted into the chat composer.
// List dialog (add / edit / delete; no reorder).
#define ID_PRESETS_DLG       1040
#define ID_PRESET_LIST       1041   // listbox of preset names
#define ID_PRESET_ADD        1042
#define ID_PRESET_EDIT       1043
#define ID_PRESET_DELETE     1044
#define ID_PRESET_HINT       1045   // "N presets" summary line (spoken)
// Edit sub-dialog: a name field + a multiline prompt body.
#define ID_PRESET_EDIT_DLG   1050
#define ID_PRE_NAME          1051   // preset name (single line)
#define ID_PRE_BODY          1052   // preset body (multiline)

#endif // RAAI_RESOURCE_H
