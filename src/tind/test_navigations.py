import tind.data.navigation.generate as generate
import tind.data.navigation.utils as utils_navigation


if __name__ == '__main__':
    for index in range(0, 12):
        print('')
        print('=' * 30)
        print(f'Navigation index: {index}')
        print('=' * 30)
        print('')

        start_node = generate.get_fixed_navigation(index=index)
        navigation_spec = start_node.navigation_spec

        utils_navigation.print_navigation(navigation_spec)