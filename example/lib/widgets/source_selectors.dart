import 'package:flutter/material.dart';

import '../sources/frame_source.dart';
import 'option_tiles.dart';

/// Source-provided chip filters and dropdowns (UVC format filter, device
/// and camera mode on Windows). Empty for the mobile camera source.
class SourceSelectors extends StatelessWidget {
  const SourceSelectors({
    super.key,
    required this.frameSource,
    required this.enabled,
  });

  final DemoFrameSource frameSource;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        for (final FrameSourceTagFilter filter in frameSource.tagFilters)
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Wrap(
                spacing: 8,
                children: [
                  for (var i = 0; i < filter.options.length; i++)
                    ChoiceChip(
                      label: Text(filter.options[i]),
                      selected: filter.selectedIndex == i,
                      onSelected: enabled ? (_) => filter.onSelect(i) : null,
                    ),
                ],
              ),
            ),
          ),
        for (final FrameSourceSelector selector in frameSource.selectors)
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
            child: LabeledDropdown<int>(
              label: selector.label,
              value: selector.selectedIndex >= 0
                  ? selector.selectedIndex
                  : null,
              isExpanded: true,
              items: [
                for (var i = 0; i < selector.options.length; i++)
                  DropdownMenuItem<int>(
                    value: i,
                    child: Text(
                      selector.options[i],
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
              ],
              onChanged: enabled
                  ? (index) {
                      if (index == null || index == selector.selectedIndex) {
                        return;
                      }
                      selector.onSelect(index);
                    }
                  : null,
            ),
          ),
      ],
    );
  }
}
