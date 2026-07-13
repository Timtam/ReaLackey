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
#define ID_PE_KEY            1028
#define ID_PE_KEYHINT        1029   // caption under the key field
#define ID_PE_MAXTURNS       1030   // "Tool steps:" (max agentic turns) field

#endif // RAAI_RESOURCE_H
