"""TLC zones inside the Manhattan Congestion Relief Zone.
Defined as Manhattan zones at or below 60th Street, per official MTA tolling rules.
Excludes Roosevelt Island and Governor's Island (geographically separate).
Verified May 8 2026 against official MTA congestion pricing map.
"""

CRZ_ZONE_IDS = [
    4,    # Alphabet City
    12,   # Battery Park
    13,   # Battery Park City
    45,   # Chinatown
    48,   # Clinton East
    68,   # East Chelsea
    79,   # East Village
    87,   # Financial District North
    88,   # Financial District South
    90,   # Flatiron
    100,  # Garment District
    107,  # Gramercy
    113,  # Greenwich Village North
    114,  # Greenwich Village South
    125,  # Hudson Sq
    137,  # Kips Bay
    144,  # Little Italy/NoLiTa
    148,  # Lower East Side
    158,  # Meatpacking/West Village West
    161,  # Midtown Center
    162,  # Midtown East
    164,  # Midtown South
    170,  # Murray Hill
    186,  # Penn Station/Madison Sq West
    209,  # Seaport
    211,  # SoHo
    224,  # Stuy Town/Peter Cooper Village
    229,  # Sutton Place/Turtle Bay North
    230,  # Times Sq/Theatre District
    231,  # TriBeCa/Civic Center
    232,  # Two Bridges/Seward Park
    233,  # UN/Turtle Bay South
    234,  # Union Sq
    246,  # West Chelsea/Hudson Yards
    249,  # West Village
    261,  # World Trade Center
]

# Zones explicitly excluded from CRZ even though geographically nearby
EXCLUDED_FROM_CRZ = [
    103, 104, 105,  # Governor's/Ellis/Liberty Islands (federal/excluded)
    202,            # Roosevelt Island (separate island, no CRZ access)
    50,             # Clinton West (Hell's Kitchen north, above 60th)
    140, 141,       # Lenox Hill (East 60s-70s)
    163,            # Midtown North (above 60th)
    237,            # Upper East Side South (60s-70s)
]